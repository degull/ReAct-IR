# e:/ReAct-IR/scripts/infer_triplet_with_metrics.py
# ------------------------------------------------------------
# Make a single triplet image: [INPUT | RESTORED | GT]
# and overlay PSNR/SSIM for (input vs gt) and (restored vs gt).
#
# Example (PowerShell):
#   python e:/ReAct-IR/scripts/infer_triplet_with_metrics.py `
#     --ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_021_L0.0204_P31.45_S0.9371.pth" `
#     --input "E:/ReAct-IR/preload_cache/CSD/000354_in.png" `
#     --gt    "E:/ReAct-IR/preload_cache/CSD/000354_gt.png" `
#     --outdir "E:/ReAct-IR/results/infer_triplets" `
#     --dim 48 --bias 1 --volterra_rank 2 --use_amp 1
#
# Batch (glob):
#   python e:/ReAct-IR/scripts/infer_triplet_with_metrics.py `
#     --ckpt "..." --input_glob "E:/ReAct-IR/preload_cache/CSD/*_in.png" `
#     --outdir "E:/ReAct-IR/results/infer_triplets" --dim 48 --bias 1 --volterra_rank 2
# ------------------------------------------------------------

import os
import sys
import glob
import math
import time
import argparse
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torch.amp import autocast

# ------------------------------------------------------------
# Make project import-safe (so "from models..." works)
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# =========================
# IO
# =========================
def imread_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def to_chw_float01(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1)

def tensor_to_u8_hwc(t_chw: torch.Tensor) -> np.ndarray:
    t = t_chw.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (t * 255.0).round().astype(np.uint8)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# Metrics (torch; no skimage dependency)
# =========================
def psnr_torch(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> float:
    """
    pred, gt: CHW or BCHW float in [0,1]
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3)).clamp_min(eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.mean().item())

def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # 1x1xKxK
    return w

def ssim_torch(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    pred, gt: CHW or BCHW float in [0,1]
    SSIM averaged over batch and channels.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    device = pred.device
    dtype = pred.dtype
    B, C, H, W = pred.shape

    # window: 1x1xKxK -> repeat over channels via groups
    win = _gaussian_window(window_size, sigma, device=device, dtype=dtype)
    win = win.repeat(C, 1, 1, 1)  # Cx1xKxK

    # conv2d expects NCHW
    mu1 = F.conv2d(pred, win, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(gt,   win, padding=window_size // 2, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, win, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt   * gt,   win, padding=window_size // 2, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred * gt,   win, padding=window_size // 2, groups=C) - mu12

    # standard SSIM constants for range [0,1]
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


# =========================
# Drawing / Overlay
# =========================
def _load_font(size: int = 18) -> ImageFont.FreeTypeFont:
    # robust fallback for Windows/Linux
    candidates = [
        "C:/Windows/Fonts/consola.ttf",   # Consolas
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        try:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_overlay(canvas: Image.Image, lines: List[str], x: int, y: int, font: ImageFont.ImageFont):
    draw = ImageDraw.Draw(canvas)
    # simple readable box
    pad = 8
    spacing = 4

    # measure
    widths = []
    heights = []
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    box_w = max(widths) + 2 * pad
    box_h = sum(heights) + (len(lines) - 1) * spacing + 2 * pad

    # background
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0, 160))
    # text
    yy = y + pad
    for ln, h in zip(lines, heights):
        draw.text((x + pad, yy), ln, fill=(255, 255, 255), font=font)
        yy += h + spacing


def make_triplet_with_text(inp_u8: np.ndarray, pred_u8: np.ndarray, gt_u8: np.ndarray,
                           text_left: List[str], text_mid: List[str], text_right: List[str]) -> Image.Image:
    h, w = inp_u8.shape[:2]
    canvas = Image.fromarray(np.zeros((h, w * 3, 3), dtype=np.uint8))
    canvas.paste(Image.fromarray(inp_u8), (0, 0))
    canvas.paste(Image.fromarray(pred_u8), (w, 0))
    canvas.paste(Image.fromarray(gt_u8), (2 * w, 0))

    font = _load_font(18)

    # overlay boxes (top-left of each panel)
    draw_overlay(canvas, text_left,  12, 12, font)
    draw_overlay(canvas, text_mid,   w + 12, 12, font)
    draw_overlay(canvas, text_right, 2 * w + 12, 12, font)

    return canvas


# =========================
# Model / CKPT
# =========================
def load_ckpt_to_model(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # could be raw state_dict
        state = ckpt
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[CKPT] load_state_dict strict=False")
        print("  missing   :", len(missing))
        print("  unexpected:", len(unexpected))
    print("[CKPT] Loaded:", ckpt_path)


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", default="")
    ap.add_argument("--gt", default="")
    ap.add_argument("--input_glob", default="")  # e.g., E:/.../*_in.png
    ap.add_argument("--outdir", default="E:/ReAct-IR/results/infer_triplets")

    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--volterra_rank", type=int, default=2)

    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--device", default="cuda")  # cuda/cpu

    return ap.parse_args()


def resolve_pairs(args) -> List[Tuple[str, str]]:
    pairs = []
    if args.input_glob:
        in_files = sorted(glob.glob(args.input_glob))
        for ip in in_files:
            base = os.path.basename(ip)
            if "_in" not in base:
                continue
            gp = ip.replace("_in.", "_gt.")
            # if extension mismatch, try search
            if not os.path.exists(gp):
                stem = os.path.splitext(ip)[0]
                key = stem[:-3] if stem.endswith("_in") else stem
                # find any *_gt.*
                cand = []
                for ext in IMG_EXTS:
                    cand += glob.glob(key + "_gt" + ext)
                gp = cand[0] if cand else ""
            if gp and os.path.exists(gp):
                pairs.append((ip, gp))
        return pairs

    if args.input and args.gt:
        return [(args.input, args.gt)]

    raise RuntimeError("Provide either (--input and --gt) or --input_glob.")


@torch.no_grad()
def infer_one(model: torch.nn.Module, inp_path: str, gt_path: str,
              device: torch.device, use_amp: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    inp_np = imread_rgb(inp_path)
    gt_np  = imread_rgb(gt_path)

    inp_t = to_chw_float01(inp_np).unsqueeze(0).to(device)
    gt_t  = to_chw_float01(gt_np).unsqueeze(0).to(device)

    with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
        pred = model(inp_t).clamp(0, 1)

    # metrics
    # metrics: compute in float32 to avoid AMP dtype mismatch
    inp_m = inp_t.float()
    gt_m  = gt_t.float()
    pred_m = pred.float()

    in_psnr = psnr_torch(inp_m, gt_m)
    in_ssim = ssim_torch(inp_m, gt_m)
    pr_psnr = psnr_torch(pred_m, gt_m)
    pr_ssim = ssim_torch(pred_m, gt_m)


    inp_u8 = tensor_to_u8_hwc(inp_t[0])
    pred_u8 = tensor_to_u8_hwc(pred[0])
    gt_u8 = tensor_to_u8_hwc(gt_t[0])

    return inp_u8, pred_u8, gt_u8, in_psnr, in_ssim, pr_psnr, pr_ssim


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    device = torch.device("cuda" if (args.device.lower() == "cuda" and torch.cuda.is_available()) else "cpu")
    print("[Device]", device)

    model = VETNet(dim=args.dim, bias=bool(args.bias), volterra_rank=args.volterra_rank).to(device)
    model.eval()
    load_ckpt_to_model(model, args.ckpt)

    pairs = resolve_pairs(args)
    print("[Pairs]", len(pairs))

    t0 = time.time()
    for i, (inp_path, gt_path) in enumerate(pairs, start=1):
        inp_u8, pred_u8, gt_u8, in_psnr, in_ssim, pr_psnr, pr_ssim = infer_one(
            model, inp_path, gt_path, device=device, use_amp=bool(args.use_amp)
        )

        left = [
            "INPUT",
            f"PSNR vs GT: {in_psnr:.2f} dB",
            f"SSIM vs GT: {in_ssim:.4f}",
        ]
        mid = [
            "RESTORED (VETNet)",
            f"PSNR vs GT: {pr_psnr:.2f} dB",
            f"SSIM vs GT: {pr_ssim:.4f}",
        ]
        right = ["GT"]

        trip = make_triplet_with_text(inp_u8, pred_u8, gt_u8, left, mid, right)

        base = os.path.basename(inp_path)
        stem = os.path.splitext(base)[0]
        out_name = f"{stem}__P{pr_psnr:.2f}_S{pr_ssim:.4f}.png"
        out_path = os.path.join(args.outdir, out_name)
        trip.save(out_path)

        print(f"[{i:04d}/{len(pairs):04d}] saved:", out_path)

    print("[Done] total_time_sec =", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()


"""
python e:/ReAct-IR/scripts/infer_triplet_with_metrics.py `
  --ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_021_L0.0204_P31.45_S0.9371.pth" `
  --input "E:/ReAct-IR/data/CSD/Train/Snow/1.tif" `
  --gt    "E:/ReAct-IR/data/CSD/Train/Gt/1.tif" `
  --outdir "E:/ReAct-IR/results/infer_triplets" `
  --dim 48 --bias 1 --volterra_rank 2 --use_amp 1

"""