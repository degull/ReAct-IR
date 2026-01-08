# models/perception/degradation_head.py
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Ensure project root in path
# --------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DWConvBlock(nn.Module):
    """Depthwise-separable conv block: cheap + strong enough."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=s, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class TinyDegradationNet(nn.Module):
    """
    Lightweight CNN classifier head.
    Outputs logits for K classes.
    Designed to run fast on 3090.
    """
    def __init__(self, num_classes: int = 5, width: int = 32, dropout: float = 0.1):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.SiLU(inplace=True),
        )
        self.b1 = DWConvBlock(w,   w*2, k=3, s=2)   # /4
        self.b2 = DWConvBlock(w*2, w*3, k=3, s=2)   # /8
        self.b3 = DWConvBlock(w*3, w*4, k=3, s=2)   # /16
        self.b4 = DWConvBlock(w*4, w*4, k=3, s=1)   # keep

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(w*4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        logits = self.fc(x)
        return logits


def softmax_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    probs: (B,K) softmax output
    return: (B,) entropy
    """
    p = probs.clamp(min=eps, max=1.0)
    ent = -(p * p.log()).sum(dim=1)
    return ent


if __name__ == "__main__":
    # quick sanity
    net = TinyDegradationNet(num_classes=5, width=32, dropout=0.1)
    x = torch.randn(2, 3, 256, 256)
    logits = net(x)
    probs = logits.softmax(dim=1)
    ent = softmax_entropy(probs)
    print("[DEBUG] logits:", logits.shape, "probs:", probs.shape, "entropy:", ent.shape)
    print("[DEBUG] OK ✅")
