# models/toolbank/toolbank.py
import os
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from models.toolbank.lora_conv2d_multi import LoRAConv2dMulti


def is_conv1x1(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and (m.kernel_size == (1, 1))


class ToolBank(nn.Module):
    """
    ToolBank wraps a backbone and injects multi-action LoRA into eligible Conv2d modules.

    Key properties:
      - backbone weights are frozen by default
      - LoRA weights are also frozen by default
      - activate(action, scale): runtime switch
      - set_trainable_action(action): enable grads only for that action's LoRA params
      - lora_state_dict_for_action(action): save ONLY that action's LoRA tensors
      - load_lora_state_dict_for_action(action, sd): official loader (Option B)
    """

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

    # ------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------
    def activate(self, action: str, scale: float):
        action = str(action)
        scale = float(scale)
        if action == "A_STOP":
            scale = 0.0
        self.active_action = action
        self.active_scale = scale
        for _, w in self._wrapped:
            w.set_active(action, scale)

    # ------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------
    def set_trainable_action(self, action: str):
        """
        Enable grads ONLY for LoRA params belonging to `action`.
        Everything else stays frozen.
        """
        action = str(action)

        # freeze all first (includes backbone params; LoRA are inside backbone modules)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # enable chosen action only
        n = 0
        for _, w in self._wrapped:
            for p in w.trainable_parameters_for_action(action):
                p.requires_grad = True
                n += p.numel()

        print(f"[ToolBank] trainable params for {action}: {n/1e6:.4f}M")

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------
    # Save / Load (OFFICIAL APIs)
    # ------------------------------------------------------------
    def lora_state_dict_for_action(self, action: str) -> Dict[str, torch.Tensor]:
        """
        Return ONLY the LoRA tensors for `action` (full key names),
        suitable for torch.save() and later load via load_lora_state_dict_for_action().
        """
        action = str(action)
        sd = self.state_dict()
        keep: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if f"lora_A.{action}." in k or f"lora_B.{action}." in k:
                keep[k] = v.detach().cpu()
        return keep

    def load_lora_state_dict_for_action(
        self,
        action: str,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Official loader: load ONLY LoRA tensors for `action`.

        - `state_dict` should contain keys from lora_state_dict_for_action(action).
        - We load using self.load_state_dict(strict=False) because this dict is partial.
        - If strict=True, we check that ALL expected LoRA keys for this action exist.
        """
        action = str(action)

        # Move tensors to correct device on-the-fly if needed.
        # (If map_location="cpu", they stay as is; later load_state_dict will copy to target device)
        sd = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                sd[k] = v.to(map_location)
            else:
                sd[k] = v

        # Compute expected keys (from current model structure)
        expected = set(self.lora_state_dict_for_action(action).keys())
        provided = set(sd.keys())

        missing_expected = sorted(list(expected - provided))
        unexpected = sorted(list(provided - expected))

        # Load partial dict (only LoRA keys)
        missing, unexpected2 = self.load_state_dict(sd, strict=False)

        # missing from load_state_dict will include all non-LoRA keys (normal). We don't use it directly.
        # We enforce strictness only on expected LoRA keys:
        if strict and len(missing_expected) > 0:
            raise RuntimeError(
                f"[ToolBank] Missing LoRA keys for action={action} (strict=True): "
                f"{missing_expected[:10]}{' ...' if len(missing_expected) > 10 else ''}"
            )

        return {
            "action": action,
            "expected_lora_keys": len(expected),
            "provided_keys": len(provided),
            "missing_expected": len(missing_expected),
            "unexpected_provided": len(unexpected),
            "note": "Loaded with strict=False (partial) and strictness applied to LoRA key coverage only.",
        }

    def load_lora_state_dict(
        self,
        action: str,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Alias for compatibility.
        """
        return self.load_lora_state_dict_for_action(
            action=action,
            state_dict=state_dict,
            strict=strict,
            map_location=map_location,
        )

    # ------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------
    def debug_summary(self) -> Dict:
        names = [n for n, _ in self._wrapped]
        return {
            "project_root": PROJECT_ROOT,
            "wrapped_layers": len(self._wrapped),
            "sample_names": names[:10],
            "active_action": self.active_action,
            "active_scale": self.active_scale,
        }


def _debug_toolbank_injection_and_load():
    """
    Debug:
      1) inject
      2) activate
      3) set_trainable_action
      4) save lora state_dict
      5) create new ToolBank
      6) load lora state_dict
      7) check output diff is ~0
    """
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
        y1 = tb(x)
    print("[DEBUG] forward OK:", tuple(y1.shape))

    tb.set_trainable_action("A_DESNOW")
    trainable_params2 = sum(p.numel() for p in tb.parameters() if p.requires_grad)
    print(f"[DEBUG] trainable_params(after set_trainable_action)={trainable_params2/1e6:.6f}M (should be > 0)")

    sd = tb.lora_state_dict_for_action("A_DESNOW")
    n_tensors = len(sd)
    n_elems = sum(v.numel() for v in sd.values())
    print(f"[DEBUG] lora_state_dict(A_DESNOW): tensors={n_tensors} elems={n_elems/1e6:.6f}M")

    # ---- load into a fresh ToolBank and compare outputs ----
    backbone2 = VETNet(dim=48, bias=True, volterra_rank=2).to(device)
    tb2 = ToolBank(backbone2, actions=actions, rank=2, alpha=1.0, wrap_only_1x1=True).to(device)
    tb2.activate("A_DESNOW", 0.8)

    info = tb2.load_lora_state_dict_for_action("A_DESNOW", sd, strict=True, map_location="cpu")
    print("[DEBUG] load_lora_state_dict_for_action info:", info)

    with torch.no_grad():
        y2 = tb2(x)

    max_diff = (y1 - y2).abs().max().item()
    print(f"[DEBUG] save->reload max_abs_diff={max_diff:.8f} (should be 0.0 if identical init & loaded)")

    # Note:
    # This diff test is meaningful only if both tb and tb2 share identical LoRA weights AND backbone weights.
    # Here backbone is randomly initialized twice, so outputs may differ unless we also copy backbone weights.
    # We therefore also copy backbone weights to make the test strict:
    tb2.backbone.load_state_dict(tb.backbone.state_dict(), strict=True)
    with torch.no_grad():
        y3 = tb2(x)
    max_diff2 = (y1 - y3).abs().max().item()
    print(f"[DEBUG] after copying backbone weights, max_abs_diff={max_diff2:.8f} (should be ~0.0)")

    print("[ToolBank][DEBUG] OK")


if __name__ == "__main__":
    _debug_toolbank_injection_and_load()
