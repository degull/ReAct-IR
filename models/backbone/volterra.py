# models/backbone/volterra.py
import torch
import torch.nn as nn

def circular_shift(x, shift_x, shift_y):
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))


class VolterraLayer2D(nn.Module):
    """
    Exact Volterra layer used in Restormer+Volterra.
    Supports:
      - lossless quadratic interaction
      - low-rank approximation (rank >= 1)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        rank=2,
        use_lossless=False,
    ):
        super().__init__()
        self.use_lossless = use_lossless
        self.rank = rank

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )

        if use_lossless:
            self.conv2 = nn.Conv2d(
                in_channels, out_channels,
                kernel_size, padding=kernel_size // 2
            )
            self.shifts = self._generate_shifts(kernel_size)
        else:
            self.W2a = nn.ModuleList([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size, padding=kernel_size // 2
                ) for _ in range(rank)
            ])
            self.W2b = nn.ModuleList([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size, padding=kernel_size // 2
                ) for _ in range(rank)
            ])

    def _generate_shifts(self, k):
        P = k // 2
        shifts = []
        for s1 in range(-P, P + 1):
            for s2 in range(-P, P + 1):
                if s1 == 0 and s2 == 0:
                    continue
                if (s1, s2) < (0, 0):
                    continue
                shifts.append((s1, s2))
        return shifts

    def forward(self, x):
        linear_term = self.conv1(x)
        quadratic_term = 0

        if self.use_lossless:
            for s1, s2 in self.shifts:
                xs = circular_shift(x, s1, s2)
                prod = torch.clamp(x * xs, -1.0, 1.0)
                quadratic_term += self.conv2(prod)
        else:
            for a, b in zip(self.W2a, self.W2b):
                qa = torch.clamp(a(x), -1.0, 1.0)
                qb = torch.clamp(b(x), -1.0, 1.0)
                quadratic_term += qa * qb

        return linear_term + quadratic_term


if __name__ == "__main__":
    x = torch.randn(1, 48, 64, 64)
    v = VolterraLayer2D(48, 48, rank=4)
    y = v(x)
    print("[DEBUG] Volterra OK:", y.shape)
