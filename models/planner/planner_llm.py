# models/planner/planner_llm.py
import torch
import torch.nn as nn
from typing import Dict, List

from models.planner.action_space import (
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DENOISE,
    A_DEHAZE, A_DEJPEG, A_HYBRID, A_STOP
)


class PlannerLLM(nn.Module):
    """
    Lightweight Planner for ReAct-IR.
    Input  : degradation state z + uncertainty u + history summary
    Output : one action token (string)

    NOTE:
    - This is NOT a generative LLM.
    - It is a policy head producing discrete action tokens.
    """

    def __init__(
        self,
        stop_threshold: float = 0.12,
        uncertainty_threshold: float = 0.35,
        debug: bool = False,
    ):
        super().__init__()
        self.stop_threshold = stop_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.debug = debug

    @torch.no_grad()
    def forward(
        self,
        z: Dict[str, float],
        uncertainty: float,
        history: List[Dict],
    ) -> str:
        """
        z: degradation probabilities (dict)
        uncertainty: scalar uncertainty
        history: list of previous step records
        """

        # 1) Uncertainty-based safe exploration
        if uncertainty > self.uncertainty_threshold:
            if self.debug:
                print("[Planner] High uncertainty → A_HYBRID")
            return A_HYBRID

        # 2) Stop condition
        if max(z.values()) < self.stop_threshold:
            if self.debug:
                print("[Planner] All degradations below threshold → A_STOP")
            return A_STOP

        # 3) Rule-based priority (논문에서 가장 설명 쉬운 방식)
        if z.get("drop", 0.0) > 0.3:
            return A_DEDROP
        if z.get("blur", 0.0) > 0.3:
            return A_DEBLUR
        if z.get("rain", 0.0) > 0.3:
            return A_DERAIN
        if z.get("noise", 0.0) > 0.3:
            return A_DENOISE
        if z.get("haze", 0.0) > 0.3:
            return A_DEHAZE
        if z.get("jpeg", 0.0) > 0.3:
            return A_DEJPEG

        # 4) Fallback
        return A_HYBRID


if __name__ == "__main__":
    planner = PlannerLLM(debug=True)
    z = {"drop": 0.62, "blur": 0.41, "noise": 0.12}
    a = planner(z, uncertainty=0.18, history=[])
    print("[DEBUG] action:", a)
