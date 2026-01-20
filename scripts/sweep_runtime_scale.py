# scripts/sweep_runtime_scale.py
# ------------------------------------------------------------
# Sweep runtime_scale for one or multiple ToolBank actions and compute PSNR/SSIM.
#
# Windows-safe:
#   - NO local/nested functions used by DataLoader workers (picklable)
#   - Robust collate: accepts PIL or Tensor outputs from dataset
#   - Optional --num_workers 0 for maximum stability
#
# Also includes:
#   - LoRA rank auto-infer from ckpt to avoid size-mismatch
#   - --quickcheck: verifies output changes with scale (single sample)
#   - --save_best: saves best-scale outputs (per action)
#
# Example:
#   python scripts/sweep_runtime_scale.py --ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth" ^
#     --data_root "E:/ReAct-IR/data" --dataset CSDDataset --split test ^
#     --action A_DEHAZE --scales "1.0,0.8,0.6,0.4,0.2" --max_items 200 ^
#     --out_dir "E:/ReAct-IR/results/scale_sweep/csd_test_dehaze" --use_amp --save_best
# ------------------------------------------------------------
import os
import sys
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast


# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Imports from project
# --------------------------------------------------
from datasets.csd import CSDDataset
from datasets.rain100 import Rain100Dataset
from datasets.raindrop_day import DayRainDropDataset
from datasets.raindrop_night import NightRainDropDataset
from datasets.reside6k import RESIDE6KDataset

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank, AdapterSpec
from models.planner.action_space import (
    A_DEDROP, A_DEBLUR, A_DESNOW, A_DERAIN, A_DEHAZE, A_STOP
)


# ============================================================
# Image/Tensor helpers (TOP-LEVEL ONLY for Windows pickling)
# ============================================================
def ensure_rgb_pil(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_chw_float01(img: Image.Image) -> torch.Tensor:
    img = ensure_rgb_pil(img)
    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    t = torch.from_numpy(arr.transpose(2, 0, 1))  # CHW
    return t.float().clamp(0, 1)


def any_to_chw_float01(x: Any) -> torch.Tensor:
    """
    Accepts:
      - PIL.Image
      - torch.Tensor (CHW or HWC or HW)
      - numpy array (HWC or HW)
    Returns:
      - torch.Tensor CHW float32 in [0,1]
    """
    if isinstance(x, Image.Image):
        return pil_to_chw_float01(x)

    if isinstance(x, torch.Tensor):
        t = x
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.ndim == 3:
            if t.shape[0] in (1, 3):
                # CHW
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
            elif t.shape[-1] in (1, 3):
                # HWC -> CHW
                t = t.permute(2, 0, 1)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
            else:
                raise ValueError(f"[any_to_chw_float01] unknown tensor image shape: {tuple(t.shape)}")
        else:
            raise ValueError(f"[any_to_chw_float01] unsupported tensor ndim: {t.ndim}")

        t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        return t.clamp(0, 1)

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"[any_to_chw_float01] unsupported numpy shape: {arr.shape}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1))
        return t.float().clamp(0, 1)

    raise TypeError(f"[any_to_chw_float01] unsupported type: {type(x)}")


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1).cpu()
    arr = (t.numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    b, c, h, w = x.shape
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    pad = (0, pw, 0, ph)
    if ph == 0 and pw == 0:
        return x, (0, 0, 0, 0)
    x = torch.nn.functional.pad(x, pad, mode="reflect")
    return x, (0, pw, 0, ph)


def unpad(x: torch.Tensor, pad):
    _, pw, _, ph = pad
    if pw == 0 and ph == 0:
        return x
    return x[..., : x.shape[-2] - ph, : x.shape[-1] - pw]


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(eps)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    pred/target: BCHW float in [0,1]
    """
    device = pred.device
    dtype = pred.dtype

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    g = gaussian_window(window_size, sigma).to(device=device, dtype=dtype)
    w2d = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    w2d = w2d.repeat(pred.shape[1], 1, 1, 1)  # (C,1,ks,ks)

    def conv(img):
        return torch.nn.functional.conv2d(img, w2d, padding=window_size // 2, groups=img.shape[1])

    mu1 = conv(pred)
    mu2 = conv(target)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = conv(pred * pred) - mu1_sq
    sigma2_sq = conv(target * target) - mu2_sq
    sigma12 = conv(pred * target) - mu12

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


# ============================================================
# Dataset builder
# ============================================================
def build_dataset(data_root: str, dataset_name: str, split: str):
    if dataset_name == "CSDDataset":
        return CSDDataset(root=data_root, split=split, transform=None, debug=False)
    if dataset_name == "DayRainDropDataset":
        return DayRainDropDataset(root=data_root, split=split, transform=None, debug=False)
    if dataset_name == "NightRainDropDataset":
        return NightRainDropDataset(root=data_root, split=split, transform=None, debug=False)
    if dataset_name == "Rain100Dataset":
        return Rain100Dataset(root=data_root, split=split, transform=None, debug=False)
    if dataset_name == "RESIDE6KDataset":
        return RESIDE6KDataset(root=data_root, split=split, transform=None, debug=False)
    raise ValueError(dataset_name)


# ============================================================
# Robust collate (PIL or Tensor)
# ============================================================
def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs: List[torch.Tensor] = []
    gts: List[torch.Tensor] = []
    metas: List[Any] = []

    for b in batch:
        xs.append(any_to_chw_float01(b["input"]))
        gts.append(any_to_chw_float01(b["gt"]))
        metas.append(b.get("meta", None))

    return {
        "input": torch.stack(xs, dim=0),
        "gt": torch.stack(gts, dim=0),
        "meta": metas,
    }


# ============================================================
# ToolBank helpers
# ============================================================
def parse_action(a: str) -> str:
    a = a.strip()
    mapping = {
        "A_STOP": A_STOP,
        "A_DEDROP": A_DEDROP,
        "A_DEBLUR": A_DEBLUR,
        "A_DERAIN": A_DERAIN,
        "A_DESNOW": A_DESNOW,
        "A_DEHAZE": A_DEHAZE,
    }
    if a in mapping:
        return mapping[a]
    raise ValueError(f"Unknown action string: {a}")


def toolbank_apply(toolbank: ToolBank, x: torch.Tensor, action: str, runtime_scale: float) -> torch.Tensor:
    """
    Prefer toolbank.apply(x, action, runtime_scale=...)
    Fallbacks for older signatures.
    """
    try:
        return toolbank.apply(x, action, runtime_scale=float(runtime_scale))
    except TypeError:
        pass

    # fallback: just apply without runtime_scale (will ignore scale)
    toolbank.activate_adapter(action)
    return toolbank.forward(x)


def infer_lora_rank_from_ckpt_state(state: Dict[str, Any], default_rank: int = 4) -> int:
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if ".lora_A." in k and k.endswith(".weight"):
            if v.ndim == 4:
                return int(v.shape[0])
            if v.ndim == 2:
                return int(v.shape[0])
    return int(default_rank)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--dataset", type=str, required=True,
        choices=["CSDDataset", "DayRainDropDataset", "NightRainDropDataset", "Rain100Dataset", "RESIDE6KDataset"]
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    ap.add_argument("--action", type=str, default="A_DEHAZE", help="e.g. A_DEHAZE or comma-separated multiple actions")
    ap.add_argument("--scales", type=str, default="1.0,0.8,0.6,0.4", help="comma-separated floats")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_items", type=int, default=0, help="0=all")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--pad_multiple", type=int, default=8)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--quickcheck", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[CKPT]", args.ckpt)

    # dataset / loader
    ds = build_dataset(args.data_root, args.dataset, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_single,
        persistent_workers=(int(args.num_workers) > 0),
    )

    # ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("toolbank", ckpt)

    inferred_rank = infer_lora_rank_from_ckpt_state(state, default_rank=4)
    print(f"[LoRA] inferred rank from ckpt = {inferred_rank}")

    # model/toolbank (rank-matched)
    backbone = VETNet(dim=48, volterra_rank=4).to(device)

    def spec_rank(r: int) -> AdapterSpec:
        return AdapterSpec(rank=r, alpha=1.0, dropout=0.0, runtime_scale=1.0, init_B_zero=True, force_nonzero_init=False)

    adapter_specs = {
        A_DEDROP: spec_rank(inferred_rank),
        A_DEBLUR: spec_rank(inferred_rank),
        A_DERAIN: spec_rank(inferred_rank),
        A_DESNOW: spec_rank(inferred_rank),
        A_DEHAZE: spec_rank(inferred_rank),
    }

    toolbank = ToolBank(backbone=backbone, adapter_specs=adapter_specs, device=device, debug=False).to(device)
    missing, unexpected = toolbank.load_state_dict(state, strict=False)
    print("[Load] missing=", len(missing), "unexpected=", len(unexpected))
    toolbank.eval()

    actions = [parse_action(x) for x in args.action.split(",") if x.strip()]
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    if len(scales) == 0:
        raise ValueError("No valid --scales provided")

    # quickcheck (one sample, two scales)
    if args.quickcheck and len(scales) >= 2:
        first = next(iter(loader))
        x0 = first["input"][0:1].to(device).clamp(0, 1)
        if args.pad_multiple > 1:
            x0p, pad = pad_to_multiple(x0, args.pad_multiple)
        else:
            x0p, pad = x0, (0, 0, 0, 0)

        a = actions[0]
        s0, s1 = scales[0], scales[-1]
        with torch.no_grad():
            if args.use_amp and device.type == "cuda":
                with autocast(device_type="cuda"):
                    y0 = toolbank_apply(toolbank, x0p, a, s0)
                    y1 = toolbank_apply(toolbank, x0p, a, s1)
            else:
                y0 = toolbank_apply(toolbank, x0p, a, s0)
                y1 = toolbank_apply(toolbank, x0p, a, s1)

        y0 = unpad(y0, pad).clamp(0, 1)
        y1 = unpad(y1, pad).clamp(0, 1)
        diff = (y0 - y1).abs().mean().item()
        print(f"[QuickCheck] action={a} mean|y(scale={s0})-y(scale={s1})| = {diff:.6e}")

    # eval
    def eval_action_scale(action: str, scale: float) -> Tuple[float, float, int]:
        ps, ss, n = 0.0, 0.0, 0
        seen = 0

        with torch.no_grad():
            for batch in loader:
                x = batch["input"].to(device, non_blocking=True).clamp(0, 1)
                gt = batch["gt"].to(device, non_blocking=True).clamp(0, 1)

                bsz = x.shape[0]
                for bi in range(bsz):
                    if args.max_items > 0 and seen >= args.max_items:
                        break

                    xi = x[bi:bi+1]
                    gti = gt[bi:bi+1]

                    if args.pad_multiple > 1:
                        xi_pad, pad = pad_to_multiple(xi, args.pad_multiple)
                        gti_pad, _ = pad_to_multiple(gti, args.pad_multiple)
                    else:
                        xi_pad, pad = xi, (0, 0, 0, 0)
                        gti_pad = gti

                    if args.use_amp and device.type == "cuda":
                        with autocast(device_type="cuda"):
                            pred = toolbank_apply(toolbank, xi_pad, action, scale)
                    else:
                        pred = toolbank_apply(toolbank, xi_pad, action, scale)

                    pred = unpad(pred, pad).clamp(0, 1)

                    ps += psnr_torch(pred.float(), gti_pad.float())
                    ss += ssim_torch(pred.float(), gti_pad.float())
                    n += 1
                    seen += 1

                if args.max_items > 0 and seen >= args.max_items:
                    break

        if n == 0:
            return 0.0, 0.0, 0
        return ps / n, ss / n, n

    # baseline
    stop_psnr, stop_ssim, n_stop = eval_action_scale(A_STOP, 0.0)
    print(f"[Baseline] {A_STOP}: PSNR={stop_psnr:.4f} SSIM={stop_ssim:.4f} (n={n_stop})")

    # sweep
    best: Dict[str, Dict[str, float]] = {}
    table: Dict[str, List[Tuple[float, float, float]]] = {}

    for act in actions:
        rows: List[Tuple[float, float, float]] = []
        for sc in scales:
            mpsnr, mssim, n = eval_action_scale(act, sc)
            rows.append((sc, mpsnr, mssim))
            print(
                f"[Sweep] {act} scale={sc:.3f} PSNR={mpsnr:.4f} SSIM={mssim:.4f}  "
                f"ΔPSNR={mpsnr-stop_psnr:+.4f} ΔSSIM={mssim-stop_ssim:+.4f} (n={n})"
            )
        table[act] = rows
        rows_sorted = sorted(rows, key=lambda x: (x[1], x[2]), reverse=True)
        best_sc, best_ps, best_ss = rows_sorted[0]
        best[act] = {"scale": float(best_sc), "psnr": float(best_ps), "ssim": float(best_ss)}

    # save summary
    summary_path = os.path.join(args.out_dir, "sweep_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"CKPT: {args.ckpt}\nDATASET: {args.dataset} split={args.split}\n")
        f.write(f"batch_size={args.batch_size} num_workers={args.num_workers} max_items={args.max_items}\n")
        f.write(f"pad_multiple={args.pad_multiple} use_amp={args.use_amp}\n\n")
        f.write(f"Baseline STOP: PSNR={stop_psnr:.6f} SSIM={stop_ssim:.6f} (n={n_stop})\n\n")
        for act in actions:
            f.write(f"Action: {act}\n")
            for sc, p, s in table[act]:
                f.write(f"  scale={sc:.3f} PSNR={p:.6f} SSIM={s:.6f} dPSNR={p-stop_psnr:+.6f} dSSIM={s-stop_ssim:+.6f}\n")
            b = best[act]
            f.write(f"  BEST: scale={b['scale']:.3f} PSNR={b['psnr']:.6f} SSIM={b['ssim']:.6f}\n\n")
    print("[Saved]", summary_path)

    # save best outputs
    if args.save_best:
        for act in actions:
            b = best[act]
            sc = float(b["scale"])
            out_dir = os.path.join(args.out_dir, f"best_{act}_scale{sc:.3f}".replace(".", "p"))
            os.makedirs(out_dir, exist_ok=True)

            seen = 0
            with torch.no_grad():
                for batch in loader:
                    x = batch["input"].to(device, non_blocking=True).clamp(0, 1)
                    bsz = x.shape[0]
                    for bi in range(bsz):
                        if args.max_items > 0 and seen >= args.max_items:
                            break
                        xi = x[bi:bi+1]
                        if args.pad_multiple > 1:
                            xi_pad, pad = pad_to_multiple(xi, args.pad_multiple)
                        else:
                            xi_pad, pad = xi, (0, 0, 0, 0)

                        if args.use_amp and device.type == "cuda":
                            with autocast(device_type="cuda"):
                                pred = toolbank_apply(toolbank, xi_pad, act, sc)
                        else:
                            pred = toolbank_apply(toolbank, xi_pad, act, sc)

                        pred = unpad(pred, pad).clamp(0, 1)
                        tensor_to_pil(pred[0]).save(os.path.join(out_dir, f"{seen:05d}.png"))
                        seen += 1
                    if args.max_items > 0 and seen >= args.max_items:
                        break

            print(f"[Saved best outputs] {act} -> {out_dir}")


if __name__ == "__main__":
    # Windows multiprocessing safety: keep entrypoint clean/top-level.
    main()
