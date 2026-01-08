# models/react_ir.py
import os
import sys
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn

# --------------------------------------------------
# Ensure project root in path
# --------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.toolbank.toolbank import (
    ToolBank,
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DEHAZE,
    A_DENOISE, A_DEJPEG, A_HYBRID, A_STOP,
)


# ============================================================
# Dummy Perception (VLM placeholder)
# ============================================================

class DummyPerception(nn.Module):
    """
    Estimate residual degradation probabilities + uncertainty.
    Replace with real VLM later.
    """
    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, float], float]:
        # Fake heuristic: use mean magnitude
        v = x.abs().mean().item()

        z = {
            "drop": min(1.0, v * 0.8),
            "blur": min(1.0, v * 0.6),
            "rain": min(1.0, v * 0.4),
            "haze": min(1.0, v * 0.3),
            "noise": min(1.0, v * 0.2),
        }
        # uncertainty ~ entropy proxy
        u = sum(z.values()) / len(z)
        return z, float(u)


# ============================================================
# Rule-based Planner (LLM placeholder)
# ============================================================

class RulePlanner(nn.Module):
    """
    Deterministic token planner.
    Later replace with small LLM.
    """
    def forward(
        self,
        z: Dict[str, float],
        history: List[Dict[str, Any]],
        tau: float = 0.4,
    ) -> str:

        # Priority-based policy
        if z.get("drop", 0) > tau:
            return A_DEDROP
        if z.get("blur", 0) > tau:
            return A_DEBLUR
        if z.get("rain", 0) > tau:
            return A_DERAIN
        if z.get("haze", 0) > tau:
            return A_DEHAZE
        if z.get("noise", 0) > tau:
            return A_DENOISE

        return A_STOP


# ============================================================
# Judge (feedback / stop decision)
# ============================================================

class SimpleJudge(nn.Module):
    """
    Decide whether to stop based on residuals.
    """
    def forward(
        self,
        z: Dict[str, float],
        u: float,
        tau_stop: float = 0.15,
    ) -> bool:
        max_r = max(z.values())
        return max_r < tau_stop


# ============================================================
# ReAct-IR Core
# ============================================================

class ReActIR(nn.Module):
    """
    ReAct-IR Agent
    """

    def __init__(
        self,
        toolbank: ToolBank,
        perception: nn.Module,
        planner: nn.Module,
        judge: nn.Module,
        max_steps: int = 5,
        tau_uncertainty: float = 0.7,
        verbose: bool = True,
    ):
        super().__init__()
        self.toolbank = toolbank
        self.perception = perception
        self.planner = planner
        self.judge = judge

        self.max_steps = max_steps
        self.tau_u = tau_uncertainty
        self.verbose = verbose

    @torch.no_grad()
    def forward(self, x0: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Run autoregressive restoration loop.
        """
        x = x0
        history: List[Dict[str, Any]] = []

        for t in range(1, self.max_steps + 1):
            z, u = self.perception(x)

            if self.verbose:
                print(f"\n[ReAct-IR] Step {t}")
                print(f"  z={z}")
                print(f"  uncertainty={u:.3f}")

            # Planner
            if u > self.tau_u:
                action = A_HYBRID
                if self.verbose:
                    print("  High uncertainty → fallback A_HYBRID")
            else:
                action = self.planner(z, history)

            if self.verbose:
                print(f"  Planner action: {action}")

            # Stop check
            if action == A_STOP:
                if self.verbose:
                    print("  Planner requested STOP")
                break

            # Execute tool
            x = self.toolbank.apply(x, action)

            # Judge
            stop_flag = self.judge(z, u)
            history.append({
                "step": t,
                "z": z,
                "u": u,
                "action": action,
                "stop": stop_flag,
            })

            if stop_flag:
                if self.verbose:
                    print("  Judge decided STOP")
                break

        return x, history


# ============================================================
# Debug / Smoke Test
# ============================================================

if __name__ == "__main__":
    from models.toolbank.toolbank import ToolBank

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 3, 3, padding=1)

        def forward(self, x):
            return self.conv2(self.conv1(x))

    print("\n[DEBUG] Initializing ReAct-IR...")

    backbone = DummyBackbone()

    action_to_targets = {
        A_DEDROP: ["conv1"],
        A_DEBLUR: ["conv2"],
        A_DERAIN: ["conv1", "conv2"],
        A_HYBRID: ["conv1", "conv2"],
    }

    toolbank = ToolBank(
        backbone=backbone,
        action_to_targets=action_to_targets,
        lora_rank=8,
        lora_alpha=8.0,
        verbose=True,
    )

    perception = DummyPerception()
    planner = RulePlanner()
    judge = SimpleJudge()

    agent = ReActIR(
        toolbank=toolbank,
        perception=perception,
        planner=planner,
        judge=judge,
        max_steps=4,
        verbose=True,
    )

    x = torch.randn(1, 3, 64, 64)
    y, trace = agent(x)

    print("\n[DEBUG] Final output shape:", y.shape)
    print("\n[DEBUG] Action trace:")
    for h in trace:
        print(h)

    print("\n[DEBUG] ReAct-IR OK ✅")
