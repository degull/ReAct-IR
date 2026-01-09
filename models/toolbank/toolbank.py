# E:\ReAct-IR\models\toolbank\toolbank.py
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
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW,
    A_DEHAZE, A_HYBRID, A_STOP
)
from models.toolbank.lora import LoRAConv2d


# --------------------------
# Action → target patterns
# --------------------------
def _patterns_for_action(action: str):
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
    if action == A_HYBRID:
        return [".attn.project_out", ".volt1."]
    return []


@dataclass
class AdapterSpec:
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    runtime_scale: float = 1.0


class ToolBank(nn.Module):
    """
    Shared-backbone + action-specific LoRA adapters
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

        # ✅ action → LoRA modules
        self.adapters: Dict[str, List[LoRAConv2d]] = {}

        self._inject_all_actions()
        self.activate_adapter(A_STOP)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _iter_named_conv_or_lora(self):
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
        🔥 DEBUG MODE: force_nonzero_init=True
        """
        lora = LoRAConv2d(
            conv,
            r=spec.rank,
            alpha=spec.alpha,
            force_nonzero_init=False,  # 🔥 핵심
        )
        lora.active = True
        return lora

    # --------------------------------------------------
    # Injection
    # --------------------------------------------------
    def _inject_action(self, action: str):
        patterns = _patterns_for_action(action)
        spec = self.adapter_specs.get(action, AdapterSpec())
        action_loras: List[LoRAConv2d] = []

        for name, module in self._iter_named_conv_or_lora():
            if not patterns:
                continue
            if not any(p in name for p in patterns):
                continue

            if isinstance(module, LoRAConv2d):
                action_loras.append(module)
                continue

            if isinstance(module, nn.Conv2d):
                lora = self._inject_lora_into_conv(module, spec)
                self._set_module_by_name(name, lora)
                action_loras.append(lora)

        # deduplicate
        uniq, seen = [], set()
        for m in action_loras:
            if id(m) not in seen:
                uniq.append(m)
                seen.add(id(m))

        self.adapters[action] = uniq

        if self.debug:
            print(
                f"[ToolBank] Action={action:<9} "
                f"Injected/Bound LoRA modules={len(uniq)} (rank={spec.rank})"
            )

    def _inject_all_actions(self):
        actions = list(self.adapter_specs.keys())
        for a in [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID]:
            if a not in actions:
                actions.append(a)
        for a in actions:
            self._inject_action(a)

    # --------------------------------------------------
    # Activation
    # --------------------------------------------------
    def activate_adapter(self, action: str):
        for a, modules in self.adapters.items():
            for m in modules:
                m.set_scale(0.0)
                m.active = False

        if action == A_STOP:
            if self.debug:
                print("[ToolBank] A_STOP → all LoRA off (no-op)")
            return

        scale = float(self.adapter_specs.get(action, AdapterSpec()).runtime_scale)
        for m in self.adapters.get(action, []):
            m.set_scale(scale)
            m.active = True

        if self.debug:
            print(
                f"[ToolBank] Activated action={action} "
                f"| scale={scale} | #modules={len(self.adapters.get(action, []))}"
            )

    # --------------------------------------------------
    # Apply
    # --------------------------------------------------
    def apply(self, x: torch.Tensor, action: str) -> torch.Tensor:
        if self.debug:
            print(f"[DEBUG] apply() using action = {action}")

        if action == A_STOP:
            self.activate_adapter(A_STOP)
            return self.backbone(x)   # ✅ backbone은 반드시 통과

        self.activate_adapter(action)
        return self.backbone(x)



# --------------------------------------------------
# Debug main (DIFF TEST)
# --------------------------------------------------
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
            A_DEDROP: AdapterSpec(rank=4),
            A_DEBLUR: AdapterSpec(rank=4),
            A_DERAIN: AdapterSpec(rank=4),
            A_DESNOW: AdapterSpec(rank=4),
            A_DEHAZE: AdapterSpec(rank=4),
            A_HYBRID: AdapterSpec(rank=2, runtime_scale=0.8),
        },
        device=device,
        debug=True,
    ).to(device)
    tb.eval()

    x = torch.randn(1, 3, 128, 128).to(device)

    with torch.no_grad():
        out1 = tb.apply(x, A_DEDROP)
        out2 = tb.apply(x, A_DEBLUR)
        out3 = tb.apply(x, A_HYBRID)

    print("drop-blur diff:", (out1 - out2).abs().mean().item())
    print("drop-hybrid diff:", (out1 - out3).abs().mean().item())

    with torch.no_grad():
        out_stop = tb.apply(out3, A_STOP)
    print("stop diff:", (out_stop - out3).abs().mean().item())

    print("[DEBUG] ToolBank + VETNet STRUCTURE OK ✅")
