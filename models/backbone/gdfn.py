# models/backbone/gdfn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GDFN(nn.Module):
    """
    Gated-DConv Feed-Forward Network (Restormer).
    """

    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2,
            kernel_size=3, padding=1,
            groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


if __name__ == "__main__":
    x = torch.randn(1, 48, 64, 64)
    g = GDFN(48, 2.66, bias=False)
    y = g(x)
    print("[DEBUG] GDFN OK:", y.shape)
