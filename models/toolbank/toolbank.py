# E:\ReAct-IR\models\toolbank\toolbank.py
import os
import sys
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    (Condition-2) Action 전문 경로 강제:
      - apply() 내부에서 action별 입력 변형을 수행하여,
        각 action adapter가 '자기 열화에 유리한 관측'을 보도록 유도.
      - 안전하게: self.training == True 일 때만 변형(훈련 전용),
        eval/inference에서는 입력을 그대로 통과.
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
    # Action-input transforms (Condition-2)
    # --------------------------------------------------
    @staticmethod
    def _depthwise_blur3x3(x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight 3x3 depthwise Gaussian-ish blur:
          [[1,2,1],[2,4,2],[1,2,1]] / 16
        """
        b, c, h, w = x.shape
        k = torch.tensor([[1.0, 2.0, 1.0],
                          [2.0, 4.0, 2.0],
                          [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype)
        k = (k / 16.0).view(1, 1, 3, 3).repeat(c, 1, 1, 1)  # (C,1,3,3)
        return F.conv2d(x, k, bias=None, stride=1, padding=1, groups=c)

    @staticmethod
    def _avgpool(x: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)

    @staticmethod
    def _clamp01_if_needed(x: torch.Tensor) -> torch.Tensor:
        # 데이터가 [0,1] 범위라고 가정하지 않고,
        # 폭주만 막기 위해 완만하게 clamp를 걸어줌(AMP 안전).
        # 강한 clamp는 학습을 망칠 수 있어 넉넉히.
        return torch.clamp(x, min=-2.0, max=2.0)

    def _action_input_transform(self, x: torch.Tensor, action: str) -> torch.Tensor:
        """
        Action별 "전문 경로"를 강제하기 위한 입력 변형.
        - 훈련 때만 적용 (self.training==True)
        - eval/infer에서는 입력 그대로 반환
        """
        if (not self.training) or (action == A_STOP):
            return x

        # 공통: AMP/안정성을 위해 dtype/device 유지, 값 폭주만 완만하게 억제
        # (여기서는 변형 후에만 clamp)
        if action == A_DEBLUR:
            # DeBlur: 약한 블러를 더해 "블러 특성"을 강조(전문 경로 유도)
            y = self._depthwise_blur3x3(x)
            out = 0.7 * x + 0.3 * y

        elif action == A_DEHAZE:
            # DeHaze: 저주파(대기광/콘트라스트 저하) 성분을 강조
            lf = self._avgpool(x, k=15)
            out = 0.6 * x + 0.4 * lf

        elif action == A_DERAIN:
            # DeRain: 스트릭/고주파 성분을 강조 (미세 high-pass)
            lf = self._avgpool(x, k=7)
            hp = x - lf
            out = x + 0.5 * hp

        elif action in (A_DEDROP, A_DESNOW):
            # DeDrop / DeSnow: 국소적인 blob/occlusion 성분을 강조 (좀 더 강한 high-pass)
            lf = self._avgpool(x, k=11)
            hp = x - lf
            out = x + 0.8 * hp

        elif action == A_HYBRID:
            # Hybrid: 저주파(안개) + 고주파(스트릭/블랍) 둘 다 보이도록 혼합
            lf = self._avgpool(x, k=15)
            mf = self._avgpool(x, k=7)
            hp = x - mf
            out = 0.55 * x + 0.25 * lf + 0.35 * hp

        else:
            out = x

        return self._clamp01_if_needed(out)

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

        # (Condition-2) action별 입력 변형 (훈련 전용)
        x_in = self._action_input_transform(x, action)

        self.activate_adapter(action)
        return self.backbone(x_in)



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
