# scripts/infer_diagnoser_maps.py
import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]

# -----------------------
# Models (same as train_diagnoser.py)
class ViTMapDiagnoser(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        num_labels=5,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
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
        self.map_head = nn.Linear(embed_dim, num_labels)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.map_head.weight, std=0.02)
        nn.init.zeros_(self.map_head.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)                  # (B,D,Hp,Wp)
        x = x.flatten(2).transpose(1, 2)         # (B,N,D)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)
        x = self.encoder(x)                      # (B,N,D)
        x = self.norm(x)

        patch_logits = self.map_head(x)          # (B,N,5)
        m0 = patch_logits.transpose(1, 2).reshape(B, 5, self.grid, self.grid)

        logit_mean = m0.mean(dim=(2, 3))
        logit_max = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)
        return logits, m0


class CNNMapDiagnoser(nn.Module):
    def __init__(self, num_labels=5, width=64):
        super().__init__()
        w = width
        self.feat = nn.Sequential(
            nn.Conv2d(3, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, 2*w, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(2*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(4*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),
        )
        self.map_head = nn.Conv2d(4*w, num_labels, 1, 1, 0)

    def forward(self, x):
        f = self.feat(x)
        m0 = self.map_head(f)
        logit_mean = m0.mean(dim=(2, 3))
        logit_max = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)
        return logits, m0


# -----------------------
def load_rgb(path, size=256):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return img, x

def save_heatmap_gray(m, out_path):
    # m: (H,W) float
    m = m - m.min()
    m = m / (m.max() + 1e-8)
    im = (m * 255.0).astype(np.uint8)
    Image.fromarray(im, mode="L").save(out_path)

def save_overlay_red(base_rgb, heat, out_path, alpha=0.5):
    # base_rgb: PIL RGB
    # heat: (H,W) in [0,1]
    base = np.array(base_rgb).astype(np.float32) / 255.0
    h = heat[..., None]  # (H,W,1)
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    over = (1 - alpha*h) * base + (alpha*h) * red
    over = np.clip(over, 0, 1)
    Image.fromarray((over * 255).astype(np.uint8)).save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", default="E:/ReAct-IR/results/diagnoser_vis")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=None, choices=[None, "vit_map", "cnn_map"])
    ap.add_argument("--img_size", type=int, default=256)

    # vit params (match training if vit_map)
    ap.add_argument("--vit_patch", type=int, default=16)
    ap.add_argument("--vit_dim", type=int, default=384)
    ap.add_argument("--vit_depth", type=int, default=6)
    ap.add_argument("--vit_heads", type=int, default=6)

    # cnn params (match training if cnn_map)
    ap.add_argument("--cnn_width", type=int, default=64)

    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.ckpt, map_location="cpu")

    # infer model type from ckpt if not provided
    model_name = a.model or ckpt.get("model", "vit_map")
    print("[CKPT model]", model_name)

    if model_name == "vit_map":
        model = ViTMapDiagnoser(
            img_size=a.img_size,
            patch_size=a.vit_patch,
            embed_dim=a.vit_dim,
            depth=a.vit_depth,
            num_heads=a.vit_heads,
            num_labels=5,
        )
    else:
        model = CNNMapDiagnoser(num_labels=5, width=a.cnn_width)

    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    assert sd is not None, "No state_dict in ckpt"
    model.load_state_dict(sd, strict=True)

    model.to(device).eval()

    base_img, x = load_rgb(a.image, size=a.img_size)
    x = x.to(device)

    with torch.no_grad():
        logits, m0 = model(x)                 # (1,5), (1,5,h,w)
        probs = torch.sigmoid(logits)[0]      # (5,)
        m0_up = F.interpolate(m0, size=(a.img_size, a.img_size), mode="bilinear", align_corners=False)[0]  # (5,H,W)

    # ---- print global logits/probs ----
    print("\n[Global logits/probs]")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:5s}: logit={logits[0,i].item():+.4f}  prob={probs[i].item():.4f}")

    # ---- save maps ----
    for i, name in enumerate(LABEL_NAMES):
        heat = m0_up[i].detach().cpu().numpy()
        # normalize to [0,1] for visualization
        h = heat - heat.min()
        h = h / (h.max() + 1e-8)

        gray_path = os.path.join(a.outdir, f"m0_{name}.png")
        over_path = os.path.join(a.outdir, f"overlay_{name}.png")
        save_heatmap_gray(h, gray_path)
        save_overlay_red(base_img, h, over_path, alpha=0.6)

    # save base too
    base_img.save(os.path.join(a.outdir, "input.png"))
    print(f"\n[Saved] {a.outdir}")
    print("  - input.png")
    print("  - m0_*.png (heatmap gray)")
    print("  - overlay_*.png (heatmap overlay)")

if __name__ == "__main__":
    main()


# python scripts/infer_diagnoser_maps.py `
#   --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" `
#   --image "E:/ReAct-IR/data/CSD/Test/Snow/1.tif" `
#   --outdir "E:/ReAct-IR/results/diagnoser_vis"

# python scripts/infer_diagnoser_map.py --input "E:/ReAct-IR/data/CSD/Test/Snow/1.tif"

# python scripts/infer_diagnoser_map.py --input "E:/ReAct-IR/data/rain100H/test/rain/norain-1.png"
