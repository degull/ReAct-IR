# models/toolbank/toolbank.py
import os
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from models.toolbank.lora_conv2d_multi import LoRAConv2dMulti


def is_conv1x1(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and (m.kernel_size == (1, 1))


class ToolBank(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        actions: List[str],
        rank: int = 2,
        alpha: float = 1.0,
        wrap_only_1x1: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.actions = list(actions)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.wrap_only_1x1 = bool(wrap_only_1x1)

        self._wrapped: List[Tuple[str, LoRAConv2dMulti]] = []
        self._inject()

        # ✅ IMPORTANT: freeze everything by default
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.active_action = "A_STOP"
        self.active_scale = 0.0
        self.activate("A_STOP", 0.0)

    def _inject(self):
        def recurse(parent: nn.Module, prefix: str = ""):
            for child_name, child in list(parent.named_children()):
                full = f"{prefix}.{child_name}" if prefix else child_name

                if isinstance(child, nn.Conv2d):
                    eligible = (not self.wrap_only_1x1) or is_conv1x1(child)
                    if eligible:
                        wrapped = LoRAConv2dMulti(
                            base=child,
                            actions=self.actions,
                            rank=self.rank,
                            alpha=self.alpha,
                        )
                        setattr(parent, child_name, wrapped)
                        self._wrapped.append((full, wrapped))
                        continue

                recurse(child, full)

        recurse(self.backbone, "")
        print(f"[ToolBank] injected LoRA into {len(self._wrapped)} convs (wrap_only_1x1={self.wrap_only_1x1})")

    def activate(self, action: str, scale: float):
        action = str(action)
        scale = float(scale)
        if action == "A_STOP":
            scale = 0.0
        self.active_action = action
        self.active_scale = scale
        for _, w in self._wrapped:
            w.set_active(action, scale)

    def set_trainable_action(self, action: str):
        action = str(action)

        # freeze all first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # enable chosen action only
        n = 0
        for _, w in self._wrapped:
            for p in w.trainable_parameters_for_action(action):
                p.requires_grad = True
                n += p.numel()

        print(f"[ToolBank] trainable params for {action}: {n/1e6:.4f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def lora_state_dict_for_action(self, action: str) -> Dict[str, torch.Tensor]:
        action = str(action)
        sd = self.state_dict()
        keep: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if f"lora_A.{action}." in k or f"lora_B.{action}." in k:
                keep[k] = v
        return keep

    def debug_summary(self) -> Dict:
        names = [n for n, _ in self._wrapped]
        return {
            "project_root": PROJECT_ROOT,
            "wrapped_layers": len(self._wrapped),
            "sample_names": names[:10],
            "active_action": self.active_action,
            "active_scale": self.active_scale,
        }


def _debug_toolbank_injection():
    from models.backbone.vetnet import VETNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] device:", device)

    backbone = VETNet(dim=48, bias=True, volterra_rank=2).to(device)

    actions = ["A_DEDROP", "A_DESNOW", "A_DERAIN", "A_DEBLUR", "A_DEHAZE"]
    tb = ToolBank(backbone, actions=actions, rank=2, alpha=1.0, wrap_only_1x1=True).to(device)

    total_params = sum(p.numel() for p in tb.parameters())
    trainable_params = sum(p.numel() for p in tb.parameters() if p.requires_grad)
    print(f"[DEBUG] total_params={total_params/1e6:.2f}M  trainable_params={trainable_params/1e6:.6f}M (expect 0.0M)")

    tb.activate("A_DESNOW", 0.8)
    print("[DEBUG] summary after activate:", tb.debug_summary())

    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = tb(x)
    print("[DEBUG] forward OK:", tuple(y.shape))

    tb.set_trainable_action("A_DESNOW")
    trainable_params2 = sum(p.numel() for p in tb.parameters() if p.requires_grad)
    print(f"[DEBUG] trainable_params(after set_trainable_action)={trainable_params2/1e6:.6f}M (should be > 0)")

    sd = tb.lora_state_dict_for_action("A_DESNOW")
    n_tensors = len(sd)
    n_elems = sum(v.numel() for v in sd.values())
    print(f"[DEBUG] lora_state_dict(A_DESNOW): tensors={n_tensors} elems={n_elems/1e6:.6f}M")


if __name__ == "__main__":
    _debug_toolbank_injection()
    print("[ToolBank][DEBUG] OK")
