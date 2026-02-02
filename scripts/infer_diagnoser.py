# E:/ReAct-IR/scripts/infer_diagnoser.py
# ------------------------------------------------------------
# Inference for Map Diagnoser (vit_map / cnn_map)
# - strict load guaranteed (same model definitions as train_diagnoser.py)
# - supports checkpoints that store 'state_dict' or 'model_state_dict'
# - prints: s0 (sigmoid probs), pred_action, and m0_mean/m0_max from map
# ------------------------------------------------------------
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


# ============================================================
# Models (EXACT SAME as train_diagnoser.py)
# ============================================================
class ViTMapDiagnoser(nn.Module):
    """
    Lightweight ViT-like map diagnoser.
    - produces patch-level logits -> m0 map (B,5,Hp,Wp)
    - global logits = 0.5*(mean_pool + max_pool)
    """

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
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
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

    def forward(self, x: torch.Tensor):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)                  # (B,D,Hp,Wp)
        x = x.flatten(2).transpose(1, 2)         # (B,N,D)

        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)

        x = self.encoder(x)                      # (B,N,D)
        x = self.norm(x)

        patch_logits = self.map_head(x)          # (B,N,5)
        m0 = patch_logits.transpose(1, 2).reshape(B, 5, self.grid, self.grid)  # (B,5,Hp,Wp)

        logit_mean = m0.mean(dim=(2, 3))
        logit_max = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)  # (B,5)

        return logits, m0


class CNNMapDiagnoser(nn.Module):
    """
    Simple CNN map diagnoser.
    - m0: (B,5,h,w)
    - logits = 0.5*(mean_pool + max_pool)
    """

    def __init__(self, num_labels: int = 5, width: int = 64):
        super().__init__()
        w = width
        self.feat = nn.Sequential(
            nn.Conv2d(3, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True),          # /2
            nn.Conv2d(w, 2*w, 3, 2, 1), nn.ReLU(inplace=True),        # /4
            nn.Conv2d(2*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /8
            nn.Conv2d(4*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /16
        )
        self.map_head = nn.Conv2d(4*w, num_labels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        f = self.feat(x)
        m0 = self.map_head(f)  # (B,5,h,w)

        logit_mean = m0.mean(dim=(2, 3))
        logit_max = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)

        return logits, m0


# ============================================================
# Helpers
# ============================================================
def load_img_tensor(path: str, patch: int = 256, crop: str = "center") -> torch.Tensor:
    """
    Returns: (1,3,patch,patch) float32 in [0,1]
    crop: 'center' or 'resize'
    """
    img = Image.open(path).convert("RGB")
    W, H = img.size

    if crop == "center":
        if H >= patch and W >= patch:
            x0 = (W - patch) // 2
            y0 = (H - patch) // 2
            img = img.crop((x0, y0, x0 + patch, y0 + patch))
        else:
            img = img.resize((patch, patch), resample=Image.BILINEAR)
    else:
        img = img.resize((patch, patch), resample=Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


def _pick_state_dict(ckpt: dict) -> dict:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    return ckpt  # fallback


def build_model_from_ckpt_cfg(ckpt: dict, override_model: str = "") -> nn.Module:
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    model_name = override_model.strip() or str(cfg.get("model", "vit_map"))

    patch = int(cfg.get("patch", 256))
    if model_name == "vit_map":
        vit_patch = int(cfg.get("vit_patch", 16))
        vit_dim = int(cfg.get("vit_dim", 384))
        vit_depth = int(cfg.get("vit_depth", 6))
        vit_heads = int(cfg.get("vit_heads", 6))
        return ViTMapDiagnoser(
            img_size=patch,
            patch_size=vit_patch,
            embed_dim=vit_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            num_labels=5,
        )
    if model_name == "cnn_map":
        cnn_width = int(cfg.get("cnn_width", 64))
        return CNNMapDiagnoser(num_labels=5, width=cnn_width)

    raise ValueError(f"Unknown model_name='{model_name}'. Expected vit_map or cnn_map.")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth")
    ap.add_argument("--input", required=True)
    ap.add_argument("--use_amp", type=int, default=1)

    ap.add_argument("--model", type=str, default="", choices=["", "vit_map", "cnn_map"])
    ap.add_argument("--patch", type=int, default=-1, help="override patch size; -1 uses ckpt cfg.patch")
    ap.add_argument("--crop", type=str, default="center", choices=["center", "resize"])

    ap.add_argument("--print_logits", type=int, default=1)
    ap.add_argument("--print_map_stats", type=int, default=1)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format unexpected: expected dict with cfg/state_dict keys.")

    cfg = ckpt.get("cfg", {})
    cfg_patch = int(cfg.get("patch", 256))
    patch = int(args.patch) if int(args.patch) > 0 else cfg_patch

    model = build_model_from_ckpt_cfg(ckpt, override_model=args.model)
    sd = _pick_state_dict(ckpt)

    # strict load guarantee
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    x = load_img_tensor(args.input, patch=patch, crop=args.crop).to(device)

    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.float16, enabled=(bool(args.use_amp) and device.type == "cuda")):
            logits, m0 = model(x)

        probs = torch.sigmoid(logits)[0].detach().float().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_action = IDX_TO_ACTION[pred_idx]

    print("[CKPT]", args.ckpt)
    print("[Model]", ckpt.get("model", cfg.get("model", "unknown")))
    print("[Patch]", patch, "| crop=", args.crop)
    print("[Input]", args.input)

    if args.print_logits:
        s0 = {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)}
        print("[s0] " + "  ".join([f"{k}={s0[k]:.3f}" for k in LABEL_NAMES]))
        print("[pred_action]", pred_action)

    if args.print_map_stats:
        m0_mean = m0.mean(dim=(2, 3))[0].detach().float().cpu().numpy()
        m0_max = m0.amax(dim=(2, 3))[0].detach().float().cpu().numpy()
        print("[m0_mean(logit)] " + "  ".join([f"{LABEL_NAMES[i]}={m0_mean[i]:.3f}" for i in range(5)]))
        print("[m0_max (logit)] " + "  ".join([f"{LABEL_NAMES[i]}={m0_max[i]:.3f}" for i in range(5)]))

        m0_sig = torch.sigmoid(m0)
        m0p_mean = m0_sig.mean(dim=(2, 3))[0].detach().float().cpu().numpy()
        m0p_max = m0_sig.amax(dim=(2, 3))[0].detach().float().cpu().numpy()
        print("[m0_mean(sig)]   " + "  ".join([f"{LABEL_NAMES[i]}={m0p_mean[i]:.3f}" for i in range(5)]))
        print("[m0_max (sig)]   " + "  ".join([f"{LABEL_NAMES[i]}={m0p_max[i]:.3f}" for i in range(5)]))


if __name__ == "__main__":
    main()


# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/CSD/Test/Snow/1.tif"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/DayRainDrop/Drop/00001/00001.png"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/DayRainDrop/Blur/00001/00001.png"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/rain100H/test/rain/norain-1.png"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/RESIDE-6K/test/hazy/0001_0.8_0.2.jpg"


# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "..." --model vit_map
