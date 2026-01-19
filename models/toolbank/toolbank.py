# models/toolbank/toolbank.py
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ------------------------------------------------------------
# Ensure project root (E:/ReAct-IR) is in sys.path
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet
from models.planner.action_space import (
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_STOP
)
from models.toolbank.lora import MultiLoRAConv2d, MultiLoRALinear


# --------------------------
# Action → target patterns
#   (HYBRID 제거)
# --------------------------
def _patterns_for_action(action: str) -> List[str]:
    if action == A_DEDROP:
        return [".volt1."]
    if action == A_DEBLUR:
        return [".attn.qkv", ".attn.project_out"]
    if action == A_DERAIN:
        return [".attn.project_out"]
    if action == A_DESNOW:
        return [".volt1."]
    if action == A_DEHAZE:
        return [".attn.project_out"]
    return []


@dataclass
class AdapterSpec:
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    runtime_scale: float = 1.0
    init_B_zero: bool = True
    force_nonzero_init: bool = False


class ToolBank(nn.Module):
    """
    ToolBank (NO HYBRID)
      - Shared backbone
      - Each target layer is wrapped ONCE with MultiLoRA* wrapper
      - Inside wrapper: action별 LoRA 파라미터가 독립 (A/B가 action마다 별도)

    Guarantee:
      - action이 바뀌면 같은 레이어라도 다른 (A,B)를 사용 => 출력이 일관되게 달라짐
      - "A_DEBLUR만 업데이트 / A_DERAIN은 그대로"가 가능
        (optimizer param group을 action별로 분리하면 됨)
      - A_STOP: 전역 LoRA off (모든 MultiLoRA*에서 action=None, scale=0)
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        adapter_specs: Optional[Dict[str, AdapterSpec]] = None,
        device: Optional[torch.device] = None,
        debug: bool = True,
        enable_linear: bool = False,  # VETNet은 대부분 conv; 필요시 True
    ):
        super().__init__()
        self.debug = bool(debug)
        self.enable_linear = bool(enable_linear)

        self.backbone = backbone if backbone is not None else VETNet()
        if device is not None:
            self.backbone = self.backbone.to(device)

        # supported actions (NO HYBRID)
        self.actions: List[str] = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE]
        self.adapter_specs: Dict[str, AdapterSpec] = adapter_specs or {}

        # action -> list of MultiLoRA modules that have this action adapter
        self.adapters: Dict[str, List[nn.Module]] = {a: [] for a in self.actions}

        # runtime
        self._active_action: str = "__INIT__"

        # inject wrappers + register which module supports which actions
        self._inject_multilora_once()

        # global stop at init (real off)
        self.activate_adapter(A_STOP)

        if self.debug:
            self._diag_counts()

    # --------------------------------------------------
    # Injection (one-time)
    # --------------------------------------------------
    def _set_module_by_name(self, name: str, new_module: nn.Module):
        parts = name.split(".")
        cur = self.backbone
        for p in parts[:-1]:
            cur = getattr(cur, p)
        setattr(cur, parts[-1], new_module)

    def _iter_named_base_layers(self):
        """
        iterate over modules that are candidates to be wrapped.
        IMPORTANT: named_modules() yields nested; we only wrap if module is Conv2d/Linear AND not already wrapped.
        """
        for name, m in self.backbone.named_modules():
            if isinstance(m, MultiLoRAConv2d) or isinstance(m, MultiLoRALinear):
                continue
            if isinstance(m, nn.Conv2d):
                yield name, m
            elif self.enable_linear and isinstance(m, nn.Linear):
                yield name, m

    def _spec_for_action(self, action: str) -> AdapterSpec:
        return self.adapter_specs.get(action, AdapterSpec())

    def _collect_actions_for_layer_name(self, layer_name: str) -> List[str]:
        """
        Which actions should exist inside this wrapper?
        If multiple actions match patterns, we include all of them => params are still independent.
        """
        acts = []
        for a in self.actions:
            pats = _patterns_for_action(a)
            if len(pats) == 0:
                continue
            if any(p in layer_name for p in pats):
                acts.append(a)
        return acts

    def _inject_multilora_once(self):
        injected = 0

        # We must be careful: wrapping changes module tree; so we iterate over a snapshot list
        candidates = list(self._iter_named_base_layers())

        for name, module in candidates:
            acts = self._collect_actions_for_layer_name(name)
            if len(acts) == 0:
                continue

            # choose "shape hyperparams" from a representative spec:
            # we allow per-action different rank/alpha, but that would require per-action shapes.
            # For strict action-independence with different ranks, you'd need a more complex design.
            # Here: we enforce that all actions that share this layer use the SAME (rank, alpha, dropout) for this layer.
            # (runtime_scale can still vary per action.)
            ref = self._spec_for_action(acts[0])
            for a in acts[1:]:
                s = self._spec_for_action(a)
                if (s.rank != ref.rank) or (float(s.alpha) != float(ref.alpha)) or (float(s.dropout) != float(ref.dropout)):
                    raise ValueError(
                        f"[ToolBank] Incompatible adapter specs for layer '{name}'. "
                        f"Actions {acts} must share the same (rank, alpha, dropout) for this layer, "
                        f"but got mismatch between {acts[0]} and {a}."
                    )

            if isinstance(module, nn.Conv2d):
                wrapped = MultiLoRAConv2d(
                    module,
                    actions=acts,
                    r=ref.rank,
                    alpha=ref.alpha,
                    dropout=ref.dropout,
                    init_B_zero=bool(ref.init_B_zero),
                    force_nonzero_init=bool(ref.force_nonzero_init),
                )
            else:
                wrapped = MultiLoRALinear(
                    module,
                    actions=acts,
                    r=ref.rank,
                    alpha=ref.alpha,
                    dropout=ref.dropout,
                    init_B_zero=bool(ref.init_B_zero),
                    force_nonzero_init=bool(ref.force_nonzero_init),
                )

            self._set_module_by_name(name, wrapped)
            injected += 1

            # register per-action module list
            for a in acts:
                self.adapters[a].append(wrapped)

        if self.debug:
            print(f"[ToolBank] MultiLoRA injected wrappers = {injected}")
            for a in self.actions:
                print(f"[ToolBank] action={a:<8} modules_with_adapter={len(self.adapters[a])}")

    # --------------------------------------------------
    # Activation (global, safe)
    # --------------------------------------------------
    def _deactivate_all_lora(self):
        for _, m in self.backbone.named_modules():
            if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
                m.set_action(None)
                m.set_scale(0.0)

    def activate_adapter(self, action: str):
        if action == self._active_action:
            return

        if action == A_STOP:
            self._deactivate_all_lora()
            self._active_action = A_STOP
            if self.debug:
                print("[ToolBank] A_STOP → deactivated ALL LoRA adapters")
            return

        if action not in self.actions:
            raise KeyError(f"[ToolBank] unknown action: {action}")

        # global set to action, but scale only for modules that have that action adapter;
        # for others, action is None (off). This guarantees strict separation.
        # (This also avoids any accidental influence from modules that don't have that action.)
        for _, m in self.backbone.named_modules():
            if not isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
                continue

            if action in getattr(m, "actions", []):
                m.set_action(action)
                # runtime_scale is action-wise; applied uniformly for this action
                sc = float(self._spec_for_action(action).runtime_scale)
                m.set_scale(sc)
            else:
                m.set_action(None)
                m.set_scale(0.0)

        self._active_action = action

        if self.debug:
            sc = float(self._spec_for_action(action).runtime_scale)
            print(f"[ToolBank] Activated action={action} | runtime_scale={sc}")

    # --------------------------------------------------
    # Params helpers (for “A_DEBLUR만 업데이트” 같은 분리 학습)
    # --------------------------------------------------
    def get_action_parameters(self, action: str) -> List[torch.nn.Parameter]:
        """
        Return ONLY parameters belonging to the given action adapters.
        (base params are frozen in wrappers; only LoRA params are returned)
        """
        if action not in self.actions:
            raise KeyError(action)

        params: List[torch.nn.Parameter] = []
        seen = set()

        for m in self.adapters.get(action, []):
            # MultiLoRA wrappers store per-action adapters in ModuleDict
            for p in m.lora_A[action].parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
            for p in m.lora_B[action].parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def get_all_lora_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        seen = set()
        for _, m in self.backbone.named_modules():
            if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
                for a in getattr(m, "actions", []):
                    for p in m.lora_A[a].parameters():
                        if id(p) not in seen:
                            params.append(p)
                            seen.add(id(p))
                    for p in m.lora_B[a].parameters():
                        if id(p) not in seen:
                            params.append(p)
                            seen.add(id(p))
        return params

    # --------------------------------------------------
    # Forward / Apply
    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def apply(self, x: torch.Tensor, action: str) -> torch.Tensor:
        if self.debug:
            print(f"[DEBUG] apply(action={action})")
        self.activate_adapter(action)
        return self.forward(x)

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    def _diag_counts(self):
        total_wrappers = 0
        total_adapters = 0
        for _, m in self.backbone.named_modules():
            if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
                total_wrappers += 1
                total_adapters += len(getattr(m, "actions", []))
        print(f"[ToolBank][Diag] total MultiLoRA wrappers in backbone = {total_wrappers}")
        print(f"[ToolBank][Diag] total action-adapters across wrappers = {total_adapters}")

    def diag_first_lora_states(self, limit: int = 10) -> List[Tuple[str, Optional[str], float, bool]]:
        out = []
        for name, m in self.backbone.named_modules():
            if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
                cur = getattr(m, "current_action", None)
                sc = float(m.scale.detach().cpu().item())
                act = bool(getattr(m, "active", False))
                out.append((name, cur, sc, act))
                if len(out) >= limit:
                    break
        return out


# --------------------------------------------------
# Debug main (NO HYBRID)
# --------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[DEBUG] Initializing VETNet backbone + ToolBank (NO HYBRID) ...")
    backbone = VETNet(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        volterra_rank=4,
    ).to(device)
    backbone.eval()

    # IMPORTANT:
    # - For “diff가 0이냐/아니냐” 즉시 확인하려면 force_nonzero_init=True로 두고 테스트.
    # - 실제 학습용은 보통 init_B_zero=True, force_nonzero_init=False가 안정적.
    specs = {
        A_DEDROP: AdapterSpec(rank=2, alpha=1.0, dropout=0.0, runtime_scale=1.0, init_B_zero=True, force_nonzero_init=True),
        A_DEBLUR: AdapterSpec(rank=2, alpha=1.0, dropout=0.0, runtime_scale=1.0, init_B_zero=True, force_nonzero_init=True),
        A_DERAIN: AdapterSpec(rank=2, alpha=1.0, dropout=0.0, runtime_scale=1.0, init_B_zero=True, force_nonzero_init=True),
        A_DESNOW: AdapterSpec(rank=2, alpha=1.0, dropout=0.0, runtime_scale=1.0, init_B_zero=True, force_nonzero_init=True),
        A_DEHAZE: AdapterSpec(rank=2, alpha=1.0, dropout=0.0, runtime_scale=1.0, init_B_zero=True, force_nonzero_init=True),
    }

    tb = ToolBank(
        backbone=backbone,
        adapter_specs=specs,
        device=device,
        debug=True,
        enable_linear=False,
    ).to(device)
    tb.eval()

    x = torch.randn(1, 3, 128, 128, device=device)

    with torch.no_grad():
        out_stop = tb.apply(x, A_STOP)
        out_drop = tb.apply(x, A_DEDROP)
        out_blur = tb.apply(x, A_DEBLUR)
        out_rain = tb.apply(x, A_DERAIN)
        out_snow = tb.apply(x, A_DESNOW)
        out_haze = tb.apply(x, A_DEHAZE)

    print("stop-drop diff:", (out_stop - out_drop).abs().mean().item())
    print("drop-blur diff:", (out_drop - out_blur).abs().mean().item())
    print("blur-rain diff:", (out_blur - out_rain).abs().mean().item())
    print("snow-haze diff:", (out_snow - out_haze).abs().mean().item())

    tb.activate_adapter(A_STOP)
    print("[SCALES][STOP] first states:", tb.diag_first_lora_states(limit=8))
    tb.activate_adapter(A_DEBLUR)
    print("[SCALES][DEBLUR] first states:", tb.diag_first_lora_states(limit=8))
    tb.activate_adapter(A_STOP)
    print("[SCALES][STOP2] first states:", tb.diag_first_lora_states(limit=8))

    # param split sanity: same wrapper, different action params => different tensor ids
    p_blur = tb.get_action_parameters(A_DEBLUR)
    p_rain = tb.get_action_parameters(A_DERAIN)
    inter = set(id(p) for p in p_blur).intersection(set(id(p) for p in p_rain))
    print(f"[ParamSplit] |DEBLUR|={len(p_blur)} |DERAIN|={len(p_rain)} shared_param_ids={len(inter)}")

    print("[DEBUG] ToolBank (NO HYBRID) OK ✅")
