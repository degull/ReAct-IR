# models/toolbank/toolbank.py
import os
import sys
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DENOISE, A_DEHAZE, A_DEJPEG, A_HYBRID, A_STOP
)
from models.toolbank.lora import LoRAConv2d


# --------------------------
# Action -> target patterns
# --------------------------
def _patterns_for_action(action: str) -> List[str]:
    """
    Return list of name-substring patterns.
    If module name contains one of patterns and module is Conv2d -> inject LoRA.
    If already LoRAConv2d, we will just "reuse" that LoRA wrapper for this action.
    """
    # NOTE: This assumes VETNet uses attribute/module names like:
    #  - block.attn.qkv, block.attn.project_out
    #  - block.ffn.project_in, block.ffn.project_out
    #  - block.volt1.*, block.volt2.* (if volterra exists)
    if action == A_DEDROP:
        return [
            ".volt1.",  # local texture / drop-related refine
        ]
    if action == A_DEBLUR:
        return [
            ".attn.qkv",
            ".attn.project_out",
        ]
    if action == A_DERAIN:
        return [
            ".attn.project_out",
        ]
    if action == A_DEHAZE:
        return [
            ".attn.project_out",
        ]
    if action == A_DENOISE:
        return [
            ".volt1.",
        ]
    if action == A_DEJPEG:
        return [
            ".ffn.project_in",
            ".ffn.project_out",
        ]
    if action == A_HYBRID:
        return [
            ".attn.project_out",
            ".volt1.",
        ]
    return []


@dataclass
class AdapterSpec:
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    # runtime scale used to turn on/off LoRA without FiLM/gating
    runtime_scale: float = 1.0


class ToolBank(nn.Module):
    """
    Key design:
      - We inject LoRA wrappers into the *shared* backbone once (wrapping Conv2d -> LoRAConv2d).
      - For each action, we store a list of LoRAConv2d modules that belong to that action.
      - At runtime, we "activate" exactly one action by setting its LoRA scales > 0 and all others to 0.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        adapter_specs: Optional[Dict[str, AdapterSpec]] = None,
        device: Optional[torch.device] = None,
        debug: bool = True,
    ):
        super().__init__()
        self.debug = debug

        self.backbone = backbone if backbone is not None else VETNet()
        if device is not None:
            self.backbone = self.backbone.to(device)

        self.adapter_specs: Dict[str, AdapterSpec] = adapter_specs or {}

        # ✅ REQUIRED STRUCTURE (1): action -> list_of_lora_modules
        self.adapters: Dict[str, List[LoRAConv2d]] = {}

        self._inject_all_actions()
        self.activate_adapter(A_STOP)  # default: all off

    # --------------------------
    # Helpers: module traversal / replacement
    # --------------------------
    def _iter_named_conv_or_lora(self):
        """
        Yield (name, module) for modules that are either nn.Conv2d or LoRAConv2d.
        We need both so that:
          - if an action wants a module already wrapped by LoRAConv2d, we can reuse it.
          - if not wrapped yet, we inject it.
        """
        for name, m in self.backbone.named_modules():
            if isinstance(m, (nn.Conv2d, LoRAConv2d)):
                yield name, m

    def _set_module_by_name(self, name: str, new_module: nn.Module):
        parts = name.split(".")
        cur = self.backbone
        for p in parts[:-1]:
            cur = getattr(cur, p)
        setattr(cur, parts[-1], new_module)

    def _inject_lora_into_conv(self, conv: nn.Conv2d, spec: AdapterSpec) -> LoRAConv2d:
        """
        Wrap Conv2d into LoRAConv2d, adapting to LoRAConv2d signature.
        """
        sig = inspect.signature(LoRAConv2d.__init__)
        kwargs = {}
        # support both r/rank naming if needed
        if "r" in sig.parameters:
            kwargs["r"] = spec.rank
        if "rank" in sig.parameters:
            kwargs["rank"] = spec.rank
        if "alpha" in sig.parameters:
            kwargs["alpha"] = spec.alpha
        if "dropout" in sig.parameters:
            kwargs["dropout"] = spec.dropout
        lora = LoRAConv2d(conv, **kwargs)

        # Make sure these attributes exist for activation-style switching
        # (Even if LoRAConv2d doesn't use them internally, they help debugging.)
        if not hasattr(lora, "active"):
            lora.active = True  # type: ignore
        return lora

    # --------------------------
    # Injection
    # --------------------------
    def _inject_action(self, action: str):
        patterns = _patterns_for_action(action)
        spec = self.adapter_specs.get(action, AdapterSpec())
        action_loras: List[LoRAConv2d] = []

        for name, module in self._iter_named_conv_or_lora():
            if not patterns:
                continue
            if not any(p in name for p in patterns):
                continue

            # If already LoRA-wrapped, just reuse it for this action
            if isinstance(module, LoRAConv2d):
                action_loras.append(module)
                continue

            # Otherwise inject new LoRA wrapper
            if isinstance(module, nn.Conv2d):
                lora = self._inject_lora_into_conv(module, spec)
                self._set_module_by_name(name, lora)
                action_loras.append(lora)

        # Deduplicate (same module could be reached multiple times in some naming cases)
        uniq = []
        seen = set()
        for m in action_loras:
            if id(m) not in seen:
                uniq.append(m)
                seen.add(id(m))

        self.adapters[action] = uniq

        if self.debug:
            print(f"[ToolBank] Action={action:<9} Injected/Bound LoRA modules={len(uniq)} (rank={spec.rank})")

    def _inject_all_actions(self):
        # Ensure we cover at least these actions if present in training/testing
        actions = list(self.adapter_specs.keys())
        for a in [A_DEDROP, A_DEBLUR, A_DERAIN, A_DENOISE, A_DEHAZE, A_DEJPEG, A_HYBRID]:
            if a not in actions:
                # If user didn't define spec for it, we still create an adapter list (may be empty)
                actions.append(a)

        # Inject/bind each action
        for a in actions:
            self._inject_action(a)

    # --------------------------
    # ✅ REQUIRED STRUCTURE (2): activation
    # --------------------------
    def activate_adapter(self, action: str):
        """
        Enable exactly one action by:
          - setting its LoRA scales to runtime_scale
          - setting all other action LoRA scales to 0
        Also sets `m.active` for clarity/debugging.
        """
        # turn everything off
        for a, modules in self.adapters.items():
            for m in modules:
                if hasattr(m, "set_scale"):
                    m.set_scale(0.0)
                # optional debug flag
                try:
                    m.active = False  # type: ignore
                except Exception:
                    pass

        # STOP => no-op: keep all off
        if action == A_STOP or action is None:
            if self.debug:
                print("[ToolBank] A_STOP → all LoRA off (no-op)")
            return

        scale = float(self.adapter_specs.get(action, AdapterSpec()).runtime_scale)
        for m in self.adapters.get(action, []):
            if hasattr(m, "set_scale"):
                m.set_scale(scale)
            try:
                m.active = True  # type: ignore
            except Exception:
                pass

        if self.debug:
            print(f"[ToolBank] Activated action={action} | scale={scale} | #modules={len(self.adapters.get(action, []))}")

    # --------------------------
    # ✅ REQUIRED STRUCTURE (3): apply
    # --------------------------
    def apply(self, x: torch.Tensor, action: str) -> torch.Tensor:
        if self.debug:
            print(f"[DEBUG] apply() using action = {action}")

        if action == A_STOP:
            if self.debug:
                print("[ToolBank] A_STOP → no-op")
            return x

        if self.debug:
            print(f"[ToolBank] Applying action: {action}")

        self.activate_adapter(action)
        y = self.backbone(x)
        return y


# --------------------------
# Debug main (includes diff test)
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[DEBUG] Initializing VETNet backbone + ToolBank ...")
    backbone = VETNet(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        volterra_rank=4,
    ).to(device)
    backbone.eval()

    tb = ToolBank(
        backbone=backbone,
        adapter_specs={
            A_DEDROP: AdapterSpec(rank=4, alpha=1.0, dropout=0.0, runtime_scale=1.0),
            A_DEBLUR: AdapterSpec(rank=4, alpha=1.0, dropout=0.0, runtime_scale=1.0),
            A_DERAIN: AdapterSpec(rank=4, alpha=1.0, dropout=0.0, runtime_scale=1.0),
            A_DENOISE: AdapterSpec(rank=4, alpha=1.0, dropout=0.0, runtime_scale=1.0),
            A_DEHAZE: AdapterSpec(rank=4, alpha=1.0, dropout=0.0, runtime_scale=1.0),
            A_DEJPEG: AdapterSpec(rank=4, alpha=1.0, dropout=0.0, runtime_scale=1.0),
            A_HYBRID: AdapterSpec(rank=2, alpha=1.0, dropout=0.0, runtime_scale=0.8),
        },
        device=device,
        debug=True,
    ).to(device)
    tb.eval()

    x = torch.randn(1, 3, 128, 128).to(device)
    print("[DEBUG] Input shape:", x.shape)

    with torch.no_grad():
        out_drop = tb.apply(x, A_DEDROP)
        out_blur = tb.apply(x, A_DEBLUR)
        out_hyb  = tb.apply(x, A_HYBRID)

        diff_db = (out_drop - out_blur).abs().mean().item()
        diff_dh = (out_drop - out_hyb).abs().mean().item()

    print(f"[DIFF] drop-blur  mean|Δ| = {diff_db:.6e}")
    print(f"[DIFF] drop-hybrid mean|Δ| = {diff_dh:.6e}")

    # sanity: stop no-op
    with torch.no_grad():
        out_stop = tb.apply(out_hyb, A_STOP)
        stop_diff = (out_stop - out_hyb).abs().mean().item()
    print(f"[DIFF] stop mean|Δ| = {stop_diff:.6e}")

    print("[DEBUG] ToolBank + VETNet OK ✅")
