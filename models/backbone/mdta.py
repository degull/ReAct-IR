# models/backbone/mdta.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Attention (Restormer).
    Exact behavior preserved.
    """

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            kernel_size=3, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


if __name__ == "__main__":
    x = torch.randn(1, 48, 64, 64)
    m = MDTA(48, 4, bias=False)
    y = m(x)
    print("[DEBUG] MDTA OK:", y.shape)
