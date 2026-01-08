# models/perception/vlm.py
import os
import sys
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

# --------------------------------------------------
# Ensure project root in path
# --------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.perception.degradation_head import TinyDegradationNet, softmax_entropy


DEG_KEYS = ["drop", "blur", "rain", "haze", "noise"]  # fixed output space


class PerceptionNet(nn.Module):
    """
    Text-free residual degradation estimator.
    Outputs:
      z: dict {drop, blur, rain, haze, noise} -> probability
      u: uncertainty scalar (entropy normalized to [0,1])
    """

    def __init__(
        self,
        width: int = 32,
        dropout: float = 0.1,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.backbone = TinyDegradationNet(num_classes=len(DEG_KEYS), width=width, dropout=dropout)
        self.temperature = float(temperature)
        self.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, float], float]:
        """
        x: (B,C,H,W), typically B=1 in inference
        returns:
          z: probabilities (python floats) for B=1
          u: uncertainty (entropy normalized)
        """
        if x.dim() != 4:
            raise ValueError("[PerceptionNet] x must be (B,C,H,W)")

        logits = self.backbone(x.to(self.device))
        logits = logits / max(1e-6, self.temperature)
        probs = torch.softmax(logits, dim=1)  # (B,K)

        # entropy uncertainty
        ent = softmax_entropy(probs)  # (B,)
        ent_max = torch.log(torch.tensor(float(len(DEG_KEYS)), device=ent.device))
        u = (ent / ent_max).clamp(0, 1)  # normalize to [0,1]

        # return only first item if B=1
        p0 = probs[0].detach().cpu().tolist()
        z = {k: float(p0[i]) for i, k in enumerate(DEG_KEYS)}
        return z, float(u[0].detach().cpu().item())

    def forward_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For training/eval:
          returns probs (B,K), u (B,)
        """
        logits = self.backbone(x.to(self.device))
        logits = logits / max(1e-6, self.temperature)
        probs = torch.softmax(logits, dim=1)
        ent = softmax_entropy(probs)
        ent_max = torch.log(torch.tensor(float(len(DEG_KEYS)), device=ent.device))
        u = (ent / ent_max).clamp(0, 1)
        return probs, u


if __name__ == "__main__":
    # quick sanity
    model = PerceptionNet(width=32, dropout=0.1, temperature=1.0)
    x = torch.randn(1, 3, 256, 256)
    z, u = model(x)
    print("[DEBUG] z:", z)
    print("[DEBUG] u:", u)
    print("[DEBUG] OK ✅")
