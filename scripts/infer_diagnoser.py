# 학습된 Diagnoser로 점수 벡터 출력
import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.amp import autocast

# label order must match train_diagnoser.py
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
IDX_TO_ACTION = {
    0: "A_DEBLUR",
    1: "A_DERAIN",
    2: "A_DESNOW",
    3: "A_DEHAZE",
    4: "A_DEDROP",
}

# ----------------------------
# Model definition (must match)
# ----------------------------
class TinyViTDiagnoser(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_labels: int = 5,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                 # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)        # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, x], dim=1)          # (B,1+N,D)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)                     # (B,5)

def load_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return ckpt

def load_img_tensor(path: str, patch: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    W, H = img.size
    # center crop or resize
    if H >= patch and W >= patch:
        x0 = (W - patch) // 2
        y0 = (H - patch) // 2
        img = img.crop((x0, y0, x0 + patch, y0 + patch))
    else:
        img = img.resize((patch, patch), resample=Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1CHW
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth")
    ap.add_argument("--input", required=True, help="image path")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--use_amp", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    model = TinyViTDiagnoser(img_size=args.patch, patch_size=16, embed_dim=384, depth=6, num_heads=6, num_labels=5)
    ckpt = load_ckpt(model, args.ckpt)
    model.to(device).eval()

    x = load_img_tensor(args.input, patch=args.patch).to(device)

    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.float16, enabled=(bool(args.use_amp) and device.type == "cuda")):
            logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    # print s0
    s0 = {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)}
    pred_idx = int(np.argmax(probs))
    pred_action = IDX_TO_ACTION[pred_idx]

    print("[Input]", args.input)
    print("[s0] " + "  ".join([f"{k}={s0[k]:.3f}" for k in LABEL_NAMES]))
    print("[pred_action]", pred_action)

if __name__ == "__main__":
    main()
