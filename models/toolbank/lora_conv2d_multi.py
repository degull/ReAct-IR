# models/toolbank/lora_conv2d_multi.py
import torch
import torch.nn as nn
from typing import List, Optional


class LoRAConv2dMulti(nn.Module):
    """
    Multi-action LoRA wrapper for Conv2d.

    Forward:
      y = base(x) + scale * (alpha/rank) * B_a(A_a(x))
    """

    def __init__(
        self,
        base: nn.Conv2d,
        actions: List[str],
        rank: int = 2,
        alpha: float = 1.0,
        enabled_actions: Optional[List[str]] = None,
    ):
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError(f"LoRAConv2dMulti expects nn.Conv2d, got: {type(base)}")

        self.base = base
        self.rank = int(rank)
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        self.alpha = float(alpha)

        # Freeze base conv
        for p in self.base.parameters():
            p.requires_grad = False

        self.actions = list(actions)
        if enabled_actions is not None:
            enabled = set(enabled_actions)
            self.actions = [a for a in self.actions if a in enabled]

        in_c = base.in_channels
        out_c = base.out_channels

        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        for a in self.actions:
            self.lora_A[a] = nn.Conv2d(in_c, self.rank, kernel_size=1, bias=False)
            self.lora_B[a] = nn.Conv2d(self.rank, out_c, kernel_size=1, bias=False)

            nn.init.kaiming_uniform_(self.lora_A[a].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[a].weight)

            # ✅ IMPORTANT: default freeze LoRA too (ToolBank will enable per action)
            for p in self.lora_A[a].parameters():
                p.requires_grad = False
            for p in self.lora_B[a].parameters():
                p.requires_grad = False

        # runtime state
        self.active_action: str = "A_STOP"
        self.active_scale: float = 0.0

    def set_active(self, action: str, scale: float):
        self.active_action = str(action)
        self.active_scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        a = self.active_action
        if (a in self.lora_A) and (self.active_scale > 0.0):
            lora = self.lora_B[a](self.lora_A[a](x))
            lora = lora * (self.alpha / max(1, self.rank)) * self.active_scale
            y = y + lora
        return y

    def trainable_parameters_for_action(self, action: str):
        action = str(action)
        params = []
        if action in self.lora_A:
            params += list(self.lora_A[action].parameters())
        if action in self.lora_B:
            params += list(self.lora_B[action].parameters())
        return params

    def count_trainable_params_for_action(self, action: str) -> int:
        return sum(p.numel() for p in self.trainable_parameters_for_action(action))


def _debug_one_layer():
    torch.manual_seed(0)
    base = nn.Conv2d(8, 8, kernel_size=1, bias=False)
    actions = ["A_DEBLUR", "A_DESNOW"]
    m = LoRAConv2dMulti(base, actions=actions, rank=2, alpha=1.0)

    x = torch.randn(1, 8, 16, 16)

    m.set_active("A_STOP", 0.0)
    y0 = m(x)
    y_base = base(x)
    max_diff0 = (y0 - y_base).abs().max().item()

    m.set_active("A_DEBLUR", 1.0)
    y1 = m(x)
    max_diff1 = (y1 - y_base).abs().max().item()

    # ✅ now LoRA params are frozen by default, so trainable should be 0 until ToolBank enables
    trainable_now = sum(p.numel() for p in m.parameters() if p.requires_grad)

    return max_diff0, max_diff1, trainable_now, m.count_trainable_params_for_action("A_DEBLUR")


if __name__ == "__main__":
    d0, d1, tr0, nA = _debug_one_layer()
    print("[LoRAConv2dMulti][DEBUG] max_diff(A_STOP vs base) =", d0)
    print("[LoRAConv2dMulti][DEBUG] max_diff(A_DEBLUR init vs base) =", d1, "(should be ~0 because B=0 init)")
    print("[LoRAConv2dMulti][DEBUG] trainable params NOW =", tr0, "(should be 0; ToolBank enables per action)")
    print("[LoRAConv2dMulti][DEBUG] params belonging to A_DEBLUR =", nA, "(exists but frozen until enabled)")
