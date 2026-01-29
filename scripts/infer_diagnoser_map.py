import os
import sys
import argparse
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
IDX_TO_ACTION = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]


# ============================================================
# Utils
# ============================================================
def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def pil_open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_open_gray(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return x


def pil_to_tensor_mask(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).contiguous()  # (1,H,W)
    return x


def tensor_to_u8_rgb(x_chw: torch.Tensor) -> np.ndarray:
    x = x_chw.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def tensor_to_u8_gray(x_hw: torch.Tensor) -> np.ndarray:
    x = x_hw.detach().float().clamp(0, 1).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def resize_to_square(img: Image.Image, size: int, is_mask: bool = False) -> Image.Image:
    if img.size[0] == size and img.size[1] == size:
        return img
    res = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((size, size), resample=res)


def infer_csd_mask_path(inp_path: str) -> Optional[str]:
    """
    CSD structure:
      .../CSD/Train/Snow/xxx.tif  -> .../CSD/Train/Mask/xxx.tif
      .../CSD/Test/Snow/xxx.tif   -> .../CSD/Test/Mask/xxx.tif
    """
    p = inp_path.replace("\\", "/")
    lower = p.lower()
    if "/csd/" not in lower:
        return None
    if "/snow/" not in lower:
        return None

    parts = p.split("/")
    idx = None
    for i in range(len(parts)):
        if parts[i].lower() == "snow":
            idx = i
    if idx is None:
        return None
    parts[idx] = "Mask"
    mask_path = "/".join(parts)

    if os.path.exists(mask_path):
        return mask_path

    base, _ = os.path.splitext(mask_path)
    for e in [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]:
        cand = base + e
        if os.path.exists(cand):
            return cand
    return None


def norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)


def overlay_heatmap_on_image(img_u8: np.ndarray, heat_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_u8: HWC uint8
    heat_u8: HW uint8 (0-255)
    Make a simple red-ish overlay without matplotlib.
    """
    h = heat_u8.astype(np.float32) / 255.0
    base = img_u8.astype(np.float32) / 255.0

    # red channel boost
    overlay = base.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + h * 0.85, 0, 1)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1 - h * 0.35), 0, 1)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1 - h * 0.35), 0, 1)

    out = (base * (1 - alpha) + overlay * alpha)
    return (out * 255.0 + 0.5).astype(np.uint8)


def binarize_mask(x01: torch.Tensor, thr: float) -> torch.Tensor:
    return (x01 >= thr).float()


def compute_iou_dice_mae(pred01: torch.Tensor, gt01: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    """
    pred01, gt01: (H,W) in [0,1]
    """
    p = binarize_mask(pred01, thr)
    g = binarize_mask(gt01, thr)
    inter = (p * g).sum().item()
    union = (p + g - p * g).sum().item()
    iou = inter / (union + 1e-6)
    dice = (2 * inter) / ((p.sum().item() + g.sum().item()) + 1e-6)
    mae = (pred01 - gt01).abs().mean().item()
    return {"iou": float(iou), "dice": float(dice), "mae": float(mae)}


# ============================================================
# Model (must match train_diagnoser_map.py)
# ============================================================
class TinyViTDiagnoserMap(nn.Module):
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
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

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
        self.head_global = nn.Linear(embed_dim, num_labels)
        self.head_map = nn.Conv2d(embed_dim, num_labels, kernel_size=1, stride=1, padding=0, bias=True)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "input must be resized/cropped to img_size"

        t = self.patch_embed(x)                           # (B,D,g,g)
        t_flat = t.flatten(2).transpose(1, 2)            # (B,N,D)

        cls = self.cls_token.expand(B, -1, -1)           # (B,1,D)
        z = torch.cat([cls, t_flat], dim=1)              # (B,1+N,D)
        z = z + self.pos_embed[:, : z.size(1)]
        z = self.pos_drop(z)

        z = self.encoder(z)                              # (B,1+N,D)

        g = self.norm(z[:, 0])                           # (B,D)
        logits_global = self.head_global(g)              # (B,5)

        tokens = z[:, 1:]                                # (B,N,D)
        tokens = tokens.transpose(1, 2).contiguous()     # (B,D,N)
        tokens = tokens.view(B, -1, self.grid, self.grid)  # (B,D,g,g)

        logits_low = self.head_map(tokens)               # (B,5,g,g)
        logits_map = F.interpolate(logits_low, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return logits_global, logits_map


def load_ckpt(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # support older "head.*" -> "head_global.*"
    if "head.weight" in sd and "head_global.weight" not in sd:
        sd2 = dict(sd)
        sd2["head_global.weight"] = sd2.pop("head.weight")
        sd2["head_global.bias"] = sd2.pop("head.bias")
        sd = sd2

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected, ckpt


# ============================================================
# Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input image path")
    ap.add_argument("--ckpt", default="E:/ReAct-IR/checkpoints/diagnoser/diagnoser_map_best.pth")
    ap.add_argument("--out_dir", default="E:/ReAct-IR/results/diagnoser_map_infer")
    ap.add_argument("--img_size", type=int, default=256)

    # model hyperparams (must match training)
    ap.add_argument("--vit_patch", type=int, default=16)
    ap.add_argument("--vit_dim", type=int, default=384)
    ap.add_argument("--vit_depth", type=int, default=6)
    ap.add_argument("--vit_heads", type=int, default=6)

    ap.add_argument("--thr_action", type=float, default=0.5, help="threshold for printing (sigmoid)")
    ap.add_argument("--mask_thr", type=float, default=0.5, help="threshold for IoU/Dice binarization")
    ap.add_argument("--save_all_labels", type=int, default=1, help="1: save heatmaps for all 5, 0: only top-1")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    ap.add_argument("--no_amp", type=int, default=0)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    safe_makedirs(args.out_dir)

    # ---- load model
    model = TinyViTDiagnoserMap(
        img_size=args.img_size,
        patch_size=args.vit_patch,
        embed_dim=args.vit_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        num_labels=len(LABEL_NAMES),
    ).to(device)
    missing, unexpected, ckpt = load_ckpt(model, args.ckpt)
    model.eval()
    print("[CKPT]", args.ckpt)
    print(f"[CKPT] loaded strict=False missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  example missing:", missing[:10])
    if len(unexpected) > 0:
        print("  example unexpected:", unexpected[:10])

    # ---- read input
    inp_path = args.input
    print("[Input]", inp_path)
    img = pil_open_rgb(inp_path)
    img_rs = resize_to_square(img, args.img_size, is_mask=False)
    x = pil_to_tensor_rgb(img_rs).unsqueeze(0).to(device)

    use_amp = (device.type == "cuda") and (args.no_amp == 0)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits_g, logits_m = model(x)  # (1,5), (1,5,H,W)

    s0 = torch.sigmoid(logits_g[0]).float().cpu()  # (5,)
    m0 = torch.sigmoid(logits_m[0]).float().cpu()  # (5,H,W)

    # ---- print s0
    s_line = "  ".join([f"{LABEL_NAMES[i]}={s0[i].item():.3f}" for i in range(5)])
    print("[s0]", s_line)

    top_idx = int(torch.argmax(s0).item())
    pred_action = IDX_TO_ACTION[top_idx]
    print("[pred_action]", pred_action)

    # ---- save base image
    base_u8 = np.array(img_rs).astype(np.uint8)
    Image.fromarray(base_u8).save(os.path.join(args.out_dir, "input_resized.png"))

    # ---- choose which labels to save
    label_indices = list(range(5)) if int(args.save_all_labels) == 1 else [top_idx]

    # ---- save heatmaps
    for i in label_indices:
        name = LABEL_NAMES[i]
        heat01 = norm01(m0[i])  # (H,W)
        heat_u8 = tensor_to_u8_gray(heat01)

        # map only
        Image.fromarray(heat_u8).save(os.path.join(args.out_dir, f"map_{name}.png"))

        # overlay
        ov = overlay_heatmap_on_image(base_u8, heat_u8, alpha=float(args.alpha))
        Image.fromarray(ov).save(os.path.join(args.out_dir, f"heatmap_{name}.png"))

    # ---- CSD mask compare (only meaningful for snow)
    mask_path = infer_csd_mask_path(inp_path)
    if mask_path is not None and os.path.exists(mask_path):
        print("[CSD Mask] found:", mask_path)
        m_img = pil_open_gray(mask_path)
        m_rs = resize_to_square(m_img, args.img_size, is_mask=True)
        gt01 = pil_to_tensor_mask(m_rs)[0].clamp(0, 1)  # (H,W)

        pred_snow = norm01(m0[2])  # snow channel index=2

        metrics = compute_iou_dice_mae(pred_snow, gt01, thr=float(args.mask_thr))
        print(f"[Mask Compare] (snow) IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  MAE={metrics['mae']:.4f}")

        # save gt mask + overlay + side-by-side
        gt_u8 = tensor_to_u8_gray(gt01)
        Image.fromarray(gt_u8).save(os.path.join(args.out_dir, "csd_mask.png"))

        pred_u8 = tensor_to_u8_gray(pred_snow)
        Image.fromarray(pred_u8).save(os.path.join(args.out_dir, "pred_snow_map.png"))

        ov_gt = overlay_heatmap_on_image(base_u8, gt_u8, alpha=float(args.alpha))
        Image.fromarray(ov_gt).save(os.path.join(args.out_dir, "mask_overlay.png"))

        ov_pred = overlay_heatmap_on_image(base_u8, pred_u8, alpha=float(args.alpha))
        Image.fromarray(ov_pred).save(os.path.join(args.out_dir, "pred_overlay.png"))

        # side by side: [input | pred | gt]
        h, w = base_u8.shape[:2]
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        canvas[:, 0:w] = base_u8
        canvas[:, w:2 * w] = ov_pred
        canvas[:, 2 * w:3 * w] = ov_gt
        Image.fromarray(canvas).save(os.path.join(args.out_dir, "mask_vs_pred.png"))
    else:
        print("[CSD Mask] not found (skip compare)")

    print("[Saved]", args.out_dir)


if __name__ == "__main__":
    main()


# python scripts/infer_diagnoser_map.py --input "E:/ReAct-IR/data/CSD/Test/Snow/1.tif"

# python scripts/infer_diagnoser_map.py --input "E:/ReAct-IR/data/rain100H/test/rain/norain-1.png"
