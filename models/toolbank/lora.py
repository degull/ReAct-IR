# models/toolbank/lora.py
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LoRA Modules
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA for nn.Linear
    """
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.is_lora = True
        self.base = base

        self.r = r
        self.alpha = alpha

        # 🔥 scale은 반드시 Tensor 🔥
        self.register_buffer("scale", torch.tensor(alpha / r))

        self.lora_A = nn.Conv2d(
            base.in_channels, r, kernel_size=1, bias=False
        )
        self.lora_B = nn.Conv2d(
            r,
            base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # freeze base conv
        for p in self.base.parameters():
            p.requires_grad = False

    def set_scale(self, s: float):
        self.scale.data = self.scale.new_tensor(s)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale



# ============================================================
# LoRA Injector
# ============================================================

class LoRAInjector:
    """
    Inject LoRA modules into a backbone.
    """

    def __init__(
        self,
        r: int = 4,
        alpha: float = 1.0,
        target_modules: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules
        self.verbose = verbose

    def _match(self, name: str) -> bool:
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def inject(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Replace target layers with LoRA wrapped layers.
        Returns dict of injected modules.
        """
        injected = {}

        for name, module in model.named_modules():
            if not self._match(name):
                continue

            parent = self._get_parent(model, name)
            if parent is None:
                continue

            key = name.split(".")[-1]

            if isinstance(module, nn.Linear):
                wrapped = LoRALinear(module, self.r, self.alpha)
                setattr(parent, key, wrapped)
                injected[name] = wrapped
                if self.verbose:
                    print(f"[LoRAInjector] Injected LoRA Linear: {name}")

            elif isinstance(module, nn.Conv2d):
                wrapped = LoRAConv2d(module, self.r, self.alpha)
                setattr(parent, key, wrapped)
                injected[name] = wrapped
                if self.verbose:
                    print(f"[LoRAInjector] Injected LoRA Conv2d: {name}")

        if self.verbose:
            print(f"[LoRAInjector] Total injected modules: {len(injected)}")

        return injected

    def _get_parent(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """
        Get parent module given full module name.
        """
        parts = name.split(".")
        cur = model
        for p in parts[:-1]:
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur


# ============================================================
# Debug / Smoke Test
# ============================================================

def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    """
    Smoke test for LoRA injection.

    This does NOT depend on VETNet.
    It tests injection on a dummy model.
    """

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.mean(dim=[2, 3])
            x = self.fc(x)
            return x

    print("\n[DEBUG] Creating DummyNet...")
    net = DummyNet()
    print(f"[DEBUG] Total params (before): {_count_total_params(net)}")
    print(f"[DEBUG] Trainable params (before): {_count_trainable_params(net)}")

    injector = LoRAInjector(
        r=8,
        alpha=8.0,
        target_modules=None,  # inject all conv/linear
        verbose=True
    )

    print("\n[DEBUG] Injecting LoRA...")
    injector.inject(net)

    print(f"[DEBUG] Total params (after): {_count_total_params(net)}")
    print(f"[DEBUG] Trainable params (after): {_count_trainable_params(net)}")

    x = torch.randn(2, 3, 64, 64)
    y = net(x)

    print(f"[DEBUG] Forward OK. Output shape: {y.shape}")
    print("[DEBUG] LoRAInjector OK ✅")
