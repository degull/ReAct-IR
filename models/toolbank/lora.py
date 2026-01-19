# models/toolbank/lora.py
import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, List, Optional, Sequence


# ============================================================
# Multi-Adapter LoRA (action-specific params, shared base)
# ============================================================

class MultiLoRALinear(nn.Module):
    """
    Multi-adapter LoRA for nn.Linear.

    y = base(x) + (scale * alpha/r) * B_action( Drop( A_action(x) ) )
    """

    def __init__(
        self,
        base: nn.Linear,
        actions: Sequence[str],
        r: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_B_zero: bool = True,
        force_nonzero_init: bool = False,
    ):
        super().__init__()
        self.is_lora = True
        self.base = base

        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)

        # runtime state
        self.current_action: Optional[str] = None
        self.active: bool = False
        self.register_buffer("scale", torch.tensor(0.0))  # runtime-controlled

        self.actions: List[str] = list(actions)

        # NOTE: use nn.ModuleDict to avoid Pylance "module used as type" issues
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_drop = nn.ModuleDict()

        for a in self.actions:
            self.lora_A[a] = nn.Linear(base.in_features, self.r, bias=False)
            self.lora_B[a] = nn.Linear(self.r, base.out_features, bias=False)
            self.lora_drop[a] = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

            nn.init.kaiming_uniform_(self.lora_A[a].weight, a=5 ** 0.5)

            if force_nonzero_init:
                nn.init.kaiming_uniform_(self.lora_B[a].weight, a=5 ** 0.5)
            else:
                if init_B_zero:
                    nn.init.zeros_(self.lora_B[a].weight)
                else:
                    nn.init.kaiming_uniform_(self.lora_B[a].weight, a=5 ** 0.5)

        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def set_action(self, action: Optional[str]) -> None:
        if action is None:
            self.current_action = None
            self.active = False
            return
        if action not in self.lora_A:
            raise KeyError(f"[MultiLoRALinear] unknown action: {action}")
        self.current_action = action
        self.active = True

    def set_scale(self, s: float) -> None:
        self.scale.data = self.scale.new_tensor(float(s))

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)

        # branch-skip: ensures inactive adapters do not participate in graph
        if (not self.active) or (self.current_action is None):
            return y
        if float(self.scale.detach().item()) == 0.0:
            return y

        a = self.current_action
        z = self.lora_A[a](x)
        z = self.lora_drop[a](z)
        z = self.lora_B[a](z)

        base_scale = self.alpha / float(self.r)
        return y + z * (self.scale * base_scale)


class MultiLoRAConv2d(nn.Module):
    """
    Multi-adapter LoRA for nn.Conv2d.

    y = base(x) + (scale * alpha/r) * B_action( Drop2d( A_action(x) ) )
    """

    def __init__(
        self,
        base: nn.Conv2d,
        actions: Sequence[str],
        r: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_B_zero: bool = True,
        force_nonzero_init: bool = False,
    ):
        super().__init__()
        self.is_lora = True
        self.base = base

        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)

        # runtime state
        self.current_action: Optional[str] = None
        self.active: bool = False
        self.register_buffer("scale", torch.tensor(0.0))  # runtime-controlled

        self.actions: List[str] = list(actions)

        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_drop = nn.ModuleDict()

        for a in self.actions:
            self.lora_A[a] = nn.Conv2d(base.in_channels, self.r, kernel_size=1, bias=False)
            self.lora_B[a] = nn.Conv2d(
                self.r,
                base.out_channels,
                kernel_size=base.kernel_size,
                stride=base.stride,
                padding=base.padding,
                dilation=base.dilation,
                bias=False,
            )
            self.lora_drop[a] = nn.Dropout2d(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

            nn.init.kaiming_uniform_(self.lora_A[a].weight, a=5 ** 0.5)

            if force_nonzero_init:
                nn.init.kaiming_uniform_(self.lora_B[a].weight, a=5 ** 0.5)
            else:
                if init_B_zero:
                    nn.init.zeros_(self.lora_B[a].weight)
                else:
                    nn.init.kaiming_uniform_(self.lora_B[a].weight, a=5 ** 0.5)

        # freeze base conv
        for p in self.base.parameters():
            p.requires_grad = False

    def set_action(self, action: Optional[str]) -> None:
        if action is None:
            self.current_action = None
            self.active = False
            return
        if action not in self.lora_A:
            raise KeyError(f"[MultiLoRAConv2d] unknown action: {action}")
        self.current_action = action
        self.active = True

    def set_scale(self, s: float) -> None:
        self.scale.data = self.scale.new_tensor(float(s))

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)

        # branch-skip: ensures inactive adapters do not participate in graph
        if (not self.active) or (self.current_action is None):
            return y
        if float(self.scale.detach().item()) == 0.0:
            return y

        a = self.current_action
        z = self.lora_A[a](x)
        z = self.lora_drop[a](z)
        z = self.lora_B[a](z)

        base_scale = self.alpha / float(self.r)
        return y + z * (self.scale * base_scale)


# ============================================================
# Injector (wrap selected modules once; register adapters inside)
# ============================================================

class MultiLoRAInjector:
    """
    Replace target Conv2d / Linear layers with MultiLoRA wrappers.
    Inject ONCE and store multiple adapters per action inside the wrapper.
    """

    def __init__(
        self,
        actions: Sequence[str],
        r: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        verbose: bool = True,
        init_B_zero: bool = True,
        force_nonzero_init: bool = False,
        enable_linear: bool = False,
    ):
        self.actions = list(actions)
        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.target_modules = target_modules
        self.verbose = bool(verbose)
        self.init_B_zero = bool(init_B_zero)
        self.force_nonzero_init = bool(force_nonzero_init)
        self.enable_linear = bool(enable_linear)

    def _match(self, name: str) -> bool:
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def inject(self, model: nn.Module) -> Dict[str, nn.Module]:
        injected: Dict[str, nn.Module] = {}

        # IMPORTANT: iterate on snapshot (tree changes during injection)
        for name, module in list(model.named_modules()):
            if not self._match(name):
                continue

            parent = self._get_parent(model, name)
            if parent is None:
                continue
            key = name.split(".")[-1]

            if isinstance(module, nn.Conv2d):
                wrapped = MultiLoRAConv2d(
                    module,
                    actions=self.actions,
                    r=self.r,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    init_B_zero=self.init_B_zero,
                    force_nonzero_init=self.force_nonzero_init,
                )
                setattr(parent, key, wrapped)
                injected[name] = wrapped
                if self.verbose:
                    print(f"[MultiLoRAInjector] Injected MultiLoRAConv2d: {name}")

            elif self.enable_linear and isinstance(module, nn.Linear):
                wrapped = MultiLoRALinear(
                    module,
                    actions=self.actions,
                    r=self.r,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    init_B_zero=self.init_B_zero,
                    force_nonzero_init=self.force_nonzero_init,
                )
                setattr(parent, key, wrapped)
                injected[name] = wrapped
                if self.verbose:
                    print(f"[MultiLoRAInjector] Injected MultiLoRALinear: {name}")

        if self.verbose:
            print(f"[MultiLoRAInjector] Total injected modules: {len(injected)}")
        return injected

    def _get_parent(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        cur: nn.Module = model
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
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x: Tensor) -> Tensor:
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.mean(dim=[2, 3])
            x = self.fc(x)
            return x

    actions = ["A_DEBLUR", "A_DERAIN"]

    print("\n[DEBUG] Creating DummyNet...")
    net = DummyNet()
    print(f"[DEBUG] Total params (before): {_count_total_params(net)}")
    print(f"[DEBUG] Trainable params (before): {_count_trainable_params(net)}")

    injector = MultiLoRAInjector(
        actions=actions,
        r=4,
        alpha=4.0,
        dropout=0.0,
        target_modules=None,
        verbose=True,
        init_B_zero=True,
        force_nonzero_init=True,  # make diffs non-zero immediately
        enable_linear=True,
    )

    print("\n[DEBUG] Injecting MultiLoRA...")
    injector.inject(net)

    # Activate one action globally
    for _, m in net.named_modules():
        if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
            m.set_action("A_DEBLUR")
            m.set_scale(1.0)

    print(f"[DEBUG] Total params (after): {_count_total_params(net)}")
    print(f"[DEBUG] Trainable params (after): {_count_trainable_params(net)}")

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_deblur = net(x)

        for _, m in net.named_modules():
            if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
                m.set_action("A_DERAIN")
                m.set_scale(1.0)
        y_derain = net(x)

        for _, m in net.named_modules():
            if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
                m.set_action(None)
                m.set_scale(0.0)
        y_stop = net(x)

    print(f"[DEBUG] diff(deblur-derain) mean: {(y_deblur - y_derain).abs().mean().item():.6f}")
    print(f"[DEBUG] diff(deblur-stop)  mean: {(y_deblur - y_stop).abs().mean().item():.6f}")
    print("[DEBUG] MultiLoRA OK ✅")
