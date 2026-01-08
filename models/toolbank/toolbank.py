# models/toolbank/toolbank.py
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import inspect
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
    """
    # NOTE: our VETNet uses these attribute names:
    #  - block.attn.qkv, block.attn.project_out
    #  - block.ffn.project_in, block.ffn.project_out
    #  - block.volt1.(conv1, W2a.*, W2b.* / conv2), block.volt2.(...)
    if action == A_DEDROP:
        return [
            ".volt1.",  # optional local texture refine
        ]
    if action == A_DEBLUR:
        return [
            ".attn.qkv", ".attn.project_out",  # strongest: attention
            # ".ffn.project_out",              # optional (uncomment if you want)
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
    if action == A_HYBRID:
        return [
            ".attn.project_out",
            ".volt1.",
        ]
    # STOP or unknown
    return []


@dataclass
class AdapterSpec:
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    # scale applied at runtime per action (keeps “no FiLM/gate” but still allows mild control)
    runtime_scale: float = 1.0


class ToolBank(nn.Module):
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

        self.adapter_specs = adapter_specs or {}
        self._action_loras: Dict[str, List[LoRAConv2d]] = {}

        self._inject_all_actions()
        self._disable_all_actions()

    # --------------------------
    # Injection helpers
    # --------------------------
    def _iter_named_convs(self):
        for name, m in self.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                yield name, m

    def _inject_lora_into_conv(self, conv: nn.Conv2d, spec: AdapterSpec) -> LoRAConv2d:
        sig = inspect.signature(LoRAConv2d.__init__)
        kwargs = {}
        if "r" in sig.parameters:
            kwargs["r"] = spec.rank
        if "alpha" in sig.parameters:
            kwargs["alpha"] = spec.alpha
        return LoRAConv2d(conv, **kwargs)

    def _set_module_by_name(self, name: str, new_module: nn.Module):
        parts = name.split(".")
        cur = self.backbone
        for p in parts[:-1]:
            cur = getattr(cur, p)
        setattr(cur, parts[-1], new_module)

    def _inject_action(self, action: str):
        patterns = _patterns_for_action(action)
        injected = []

        spec = self.adapter_specs.get(action, AdapterSpec())

        for name, conv in self._iter_named_convs():
            if any(p in name for p in patterns):
                if isinstance(conv, LoRAConv2d):
                    continue
                lora = self._inject_lora_into_conv(conv, spec)
                self._set_module_by_name(name, lora)
                injected.append(lora)

        self._action_loras[action] = injected
        if self.debug:
            print(f"[ToolBank] Action={action:<10} Injected LoRA modules={len(injected)} (rank={spec.rank})")

    def _inject_all_actions(self):
        actions = list(self.adapter_specs.keys())
        if A_HYBRID not in actions:
            actions.append(A_HYBRID)
        for a in actions:
            self._inject_action(a)

    # --------------------------
    # Enable / Disable
    # --------------------------
    def _disable_all_actions(self):
        for loras in self._action_loras.values():
            for m in loras:
                m.set_scale(0.0)

    def _enable_action(self, action: str):
        self._disable_all_actions()
        scale = float(self.adapter_specs.get(action, AdapterSpec()).runtime_scale)
        for m in self._action_loras.get(action, []):
            m.set_scale(scale)

    # --------------------------
    # Public API
    # --------------------------
    def apply(self, x: torch.Tensor, action: str) -> torch.Tensor:
        if action == A_STOP:
            if self.debug:
                print("[ToolBank] A_STOP → no-op")
            return x

        if self.debug:
            print(f"[ToolBank] Applying action: {action}")

        self._enable_action(action)
        y = self.backbone(x)
        return y


# --------------------------
# Debug main
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
            A_HYBRID: AdapterSpec(rank=2, alpha=1.0, dropout=0.0, runtime_scale=0.8),
        },
        device=device,
        debug=True,
    ).to(device)
    tb.eval()

    x = torch.randn(1, 3, 128, 128).to(device)
    print("[DEBUG] Input shape:", x.shape)

    y1 = tb.apply(x, A_DEDROP)
    print("[DEBUG] After A_DEDROP:", y1.shape)

    y2 = tb.apply(y1, A_DEBLUR)
    print("[DEBUG] After A_DEBLUR:", y2.shape)

    y3 = tb.apply(y2, A_HYBRID)
    print("[DEBUG] After A_HYBRID:", y3.shape)

    y4 = tb.apply(y3, A_STOP)
    print("[DEBUG] After A_STOP:", y4.shape)

    print("[DEBUG] ToolBank + VETNet OK ✅")
