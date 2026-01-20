# scripts/test_toolbank_actions.py
# ------------------------------------------------------------
# Evaluate ToolBank actions on a dataset:
#   - Baseline A_STOP vs each action (A_DEDROP/A_DEBLUR/A_DERAIN/A_DESNOW/A_DEHAZE)
#   - Compute PSNR/SSIM
#   - Optionally save restored outputs (and optionally input/gt) per action
#
# Windows-safe:
#   - No non-picklable local functions for DataLoader workers
#   - Robust collate that accepts either Tensor or PIL Image
#   - num_workers=0 safe default (recommended on Windows); you can increase after verifying
#
# Example:
#   python scripts/test_toolbank_actions.py ^
#     --ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth" ^
#     --data_root "E:/ReAct-IR/data" ^
#     --dataset "CSDDataset" --split test ^
#     --out_dir "E:/ReAct-IR/results/toolbank_eval/csd_test_actions" ^
#     --batch_size 1 --num_workers 0 --max_items 200 ^
#     --use_amp --runtime_scale 1.0 --pad_multiple 8 ^
#     --save_inputs --save_gt
#
# Notes:
#   - This script tries to infer LoRA rank from checkpoint and build VETNet accordingly.
#   - Assumes your ToolBank supports apply(x, action, runtime_scale=...)
#     (based on your recent ToolBank runtime_scale override test).
# ------------------------------------------------------------

import os
import sys
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
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
    A_STOP, A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE
)


# ============================================================
# I/O helpers
# ============================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> float tensor in [0,1], shape [3,H,W]."""
    if not isinstance(img, Image.Image):
        raise TypeError(f"pil_to_tensor expects PIL.Image, got {type(img)}")
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr.transpose(2, 0, 1))
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """float tensor [3,H,W] in [0,1] -> PIL RGB."""
    if t.ndim == 4:
        t = t[0]
    t = t.detach().clamp(0, 1).cpu()
    arr = (t.numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad [B,C,H,W] reflect to multiple. returns (x_pad, (pl, pr, pt, pb) in torch.pad order?)"""
    b, c, h, w = x.shape
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    # torch.nn.functional.pad uses (left,right,top,bottom)
    pad = (0, pw, 0, ph)
    if ph == 0 and pw == 0:
        return x, (0, 0, 0, 0)
    x = F.pad(x, pad, mode="reflect")
    return x, (0, pw, 0, ph)


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    _, pw, _, ph = pad
    if pw == 0 and ph == 0:
        return x
    return x[..., : x.shape[-2] - ph, : x.shape[-1] - pw]


# ============================================================
# Metrics
# ============================================================

def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.item())


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    return w2d


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    # pred/target: [1,3,H,W] or [B,3,H,W] ; compute mean over batch+channels
    device = pred.device
    dtype = pred.dtype
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    w = _gaussian_window(window_size, sigma, device, dtype)
    # apply per-channel (groups=channels)
    ch = pred.shape[1]
    w = w.expand(ch, 1, window_size, window_size).contiguous()

    def conv(img: torch.Tensor) -> torch.Tensor:
        return F.conv2d(img, w, padding=window_size // 2, groups=ch)

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
# Dataset builder (no local transform closures -> Windows safe)
# ============================================================

class TensorizeWrapper(torch.utils.data.Dataset):
    """
    Wrap a dataset that returns:
      - dict with keys input/gt/meta
    where input/gt may be PIL or Tensor.
    Convert to Tensor float [0,1].
    """
    def __init__(self, base_ds: torch.utils.data.Dataset):
        self.base = base_ds

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        inp = sample["input"]
        gt = sample["gt"]

        if isinstance(inp, Image.Image):
            inp_t = pil_to_tensor(inp)
        elif torch.is_tensor(inp):
            inp_t = inp.float()
        else:
            raise TypeError(f"Unknown input type: {type(inp)}")

        if isinstance(gt, Image.Image):
            gt_t = pil_to_tensor(gt)
        elif torch.is_tensor(gt):
            gt_t = gt.float()
        else:
            raise TypeError(f"Unknown gt type: {type(gt)}")

        # Ensure [0,1]
        inp_t = inp_t.clamp(0, 1)
        gt_t = gt_t.clamp(0, 1)

        return {
            "input": inp_t,
            "gt": gt_t,
            "meta": sample.get("meta", {"index": idx}),
        }


def build_dataset(data_root: str, dataset_name: str, split: str) -> torch.utils.data.Dataset:
    if dataset_name == "CSDDataset":
        ds = CSDDataset(root=data_root, split=split, transform=None, debug=False)
    elif dataset_name == "DayRainDropDataset":
        ds = DayRainDropDataset(root=data_root, split=split, transform=None, debug=False)
    elif dataset_name == "NightRainDropDataset":
        ds = NightRainDropDataset(root=data_root, split=split, transform=None, debug=False)
    elif dataset_name == "Rain100Dataset":
        ds = Rain100Dataset(root=data_root, split=split, transform=None, debug=False)
    elif dataset_name == "RESIDE6KDataset":
        ds = RESIDE6KDataset(root=data_root, split=split, transform=None, debug=False)
    else:
        raise ValueError(dataset_name)

    return TensorizeWrapper(ds)


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch elements are already tensors via TensorizeWrapper
    return {
        "input": torch.stack([b["input"] for b in batch], dim=0),
        "gt": torch.stack([b["gt"] for b in batch], dim=0),
        "meta": [b.get("meta", {}) for b in batch],
    }


# ============================================================
# ToolBank helpers
# ============================================================

def infer_lora_rank_from_ckpt(ckpt_obj: Dict[str, Any]) -> Optional[int]:
    """
    Infer rank r from any lora_A weight:
      - Conv2d: [r, in_ch, 1, 1]
      - Linear: [r, in_features]
    """
    state = ckpt_obj.get("toolbank", ckpt_obj)
    if not isinstance(state, dict):
        return None

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if ".lora_A." in k and k.endswith(".weight"):
            shape = list(v.shape)
            if len(shape) == 4:
                return int(shape[0])
            if len(shape) == 2:
                return int(shape[0])
    return None


def toolbank_apply(tb: ToolBank, x: torch.Tensor, action: str, runtime_scale: float) -> torch.Tensor:
    """
    Prefer ToolBank.apply(x, action, runtime_scale=...) if supported.
    Fallback to setting runtime_scale in tb.adapter_specs and calling apply(x, action).
    """
    try:
        return tb.apply(x, action, runtime_scale=runtime_scale)
    except TypeError:
        pass

    # fallback: override spec runtime_scale
    if hasattr(tb, "adapter_specs") and isinstance(tb.adapter_specs, dict) and action in tb.adapter_specs:
        try:
            tb.adapter_specs[action].runtime_scale = float(runtime_scale)
        except Exception:
            pass
    return tb.apply(x, action)


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CSDDataset", "DayRainDropDataset", "NightRainDropDataset", "Rain100Dataset", "RESIDE6KDataset"],
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0, help="Windows 추천: 0부터 시작")
    ap.add_argument("--max_items", type=int, default=0, help="0이면 전체")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--runtime_scale", type=float, default=1.0)
    ap.add_argument("--pad_multiple", type=int, default=8)

    ap.add_argument("--save_outputs", action="store_true", help="복원 결과 저장(액션별)")
    ap.add_argument("--save_inputs", action="store_true")
    ap.add_argument("--save_gt", action="store_true")

    ap.add_argument("--actions", type=str, default="ALL",
                    help='Comma-separated e.g. "A_STOP,A_DEHAZE". Default ALL => STOP + all actions')

    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[CKPT]", args.ckpt)

    # Dataset
    ds = build_dataset(args.data_root, args.dataset, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_single,
        persistent_workers=(args.num_workers > 0),
    )

    # Load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Infer rank and build model
    inferred_r = infer_lora_rank_from_ckpt(ckpt)
    if inferred_r is None:
        inferred_r = 2  # safe fallback for your current training
        print("[LoRA] inferred rank from ckpt = None -> fallback", inferred_r)
    else:
        print("[LoRA] inferred rank from ckpt =", inferred_r)

    backbone = VETNet(dim=48, volterra_rank=4).to(device)

    # Build ToolBank with per-action specs using inferred rank (shape must match ckpt)
    def spec():
        return AdapterSpec(rank=int(inferred_r), alpha=1.0, dropout=0.0, runtime_scale=float(args.runtime_scale),
                           init_B_zero=True, force_nonzero_init=False)

    adapter_specs = {
        A_DEDROP: spec(),
        A_DEBLUR: spec(),
        A_DERAIN: spec(),
        A_DESNOW: spec(),
        A_DEHAZE: spec(),
    }

    tb = ToolBank(backbone=backbone, adapter_specs=adapter_specs, device=device, debug=False).to(device)

    state = ckpt.get("toolbank", ckpt)
    missing, unexpected = tb.load_state_dict(state, strict=False)
    print("[Load] missing=", len(missing), "unexpected=", len(unexpected))

    # Decide actions
    all_actions = [A_STOP, A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE]
    if args.actions.strip().upper() == "ALL":
        actions = all_actions
    else:
        name_to_action = {
            "A_STOP": A_STOP,
            "A_DEDROP": A_DEDROP,
            "A_DEBLUR": A_DEBLUR,
            "A_DERAIN": A_DERAIN,
            "A_DESNOW": A_DESNOW,
            "A_DEHAZE": A_DEHAZE,
        }
        actions = []
        for tok in args.actions.split(","):
            t = tok.strip()
            if t not in name_to_action:
                raise ValueError(f"Unknown action token: {t}")
            actions.append(name_to_action[t])

    # Output dirs
    out_action_dir = {}
    if args.save_outputs or args.save_inputs or args.save_gt:
        for act in actions:
            d = os.path.join(args.out_dir, act)
            ensure_dir(d)
            out_action_dir[act] = d
            if args.save_outputs:
                ensure_dir(os.path.join(d, "pred"))
            if args.save_inputs:
                ensure_dir(os.path.join(d, "input"))
            if args.save_gt:
                ensure_dir(os.path.join(d, "gt"))

    # Evaluate
    tb.eval()

    def run_eval(act: str) -> Tuple[float, float, int]:
        ps_sum, ss_sum, n = 0.0, 0.0, 0
        seen = 0

        with torch.no_grad():
            for batch in loader:
                x = batch["input"].to(device, non_blocking=True).clamp(0, 1)
                gt = batch["gt"].to(device, non_blocking=True).clamp(0, 1)
                bsz = x.shape[0]

                for bi in range(bsz):
                    if args.max_items > 0 and seen >= args.max_items:
                        break

                    xi = x[bi:bi + 1]
                    gti = gt[bi:bi + 1]

                    if args.pad_multiple and args.pad_multiple > 1:
                        xi_pad, pad = pad_to_multiple(xi, args.pad_multiple)
                        gti_pad, _ = pad_to_multiple(gti, args.pad_multiple)
                    else:
                        xi_pad, pad = xi, (0, 0, 0, 0)
                        gti_pad = gti

                    if act == A_STOP:
                        # STOP ignores runtime_scale
                        if args.use_amp and device.type == "cuda":
                            with autocast(device_type="cuda"):
                                pred = toolbank_apply(tb, xi_pad, A_STOP, runtime_scale=0.0)
                        else:
                            pred = toolbank_apply(tb, xi_pad, A_STOP, runtime_scale=0.0)
                    else:
                        if args.use_amp and device.type == "cuda":
                            with autocast(device_type="cuda"):
                                pred = toolbank_apply(tb, xi_pad, act, runtime_scale=float(args.runtime_scale))
                        else:
                            pred = toolbank_apply(tb, xi_pad, act, runtime_scale=float(args.runtime_scale))

                    pred = unpad(pred, pad).clamp(0, 1)

                    ps_sum += psnr_torch(pred.float(), gti.float())
                    ss_sum += ssim_torch(pred.float(), gti.float())
                    n += 1

                    # Save images
                    if (args.save_outputs or args.save_inputs or args.save_gt) and act in out_action_dir:
                        base_dir = out_action_dir[act]
                        if args.save_outputs:
                            tensor_to_pil(pred[0]).save(os.path.join(base_dir, "pred", f"{seen:05d}.png"))
                        if args.save_inputs:
                            tensor_to_pil(xi[0]).save(os.path.join(base_dir, "input", f"{seen:05d}.png"))
                        if args.save_gt:
                            tensor_to_pil(gti[0]).save(os.path.join(base_dir, "gt", f"{seen:05d}.png"))

                    seen += 1

                if args.max_items > 0 and seen >= args.max_items:
                    break

        if n == 0:
            return 0.0, 0.0, 0
        return ps_sum / n, ss_sum / n, n

    results: Dict[str, Tuple[float, float, int]] = {}
    for act in actions:
        mpsnr, mssim, n = run_eval(act)
        results[act] = (mpsnr, mssim, n)
        print(f"[Eval] {act}: PSNR={mpsnr:.4f} SSIM={mssim:.4f} (n={n})")

    # Report delta vs STOP
    if A_STOP in results:
        stop_psnr, stop_ssim, _ = results[A_STOP]
        for act in actions:
            if act == A_STOP:
                continue
            p, s, n = results[act]
            print(f"[Δ vs STOP] {act}: ΔPSNR={p-stop_psnr:+.4f} ΔSSIM={s-stop_ssim:+.4f} (n={n})")

    # Save summary
    summary_path = os.path.join(args.out_dir, "actions_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"CKPT: {args.ckpt}\n")
        f.write(f"DATASET: {args.dataset} split={args.split}\n")
        f.write(f"runtime_scale={args.runtime_scale}\n")
        f.write(f"max_items={args.max_items}\n")
        f.write(f"batch_size={args.batch_size} num_workers={args.num_workers}\n")
        f.write(f"pad_multiple={args.pad_multiple} use_amp={args.use_amp}\n")
        f.write("\n")
        for act in actions:
            p, s, n = results[act]
            f.write(f"{act}: PSNR={p:.6f} SSIM={s:.6f} n={n}\n")

        if A_STOP in results:
            stop_psnr, stop_ssim, _ = results[A_STOP]
            f.write("\nDelta vs STOP:\n")
            for act in actions:
                if act == A_STOP:
                    continue
                p, s, n = results[act]
                f.write(f"{act}: dPSNR={p-stop_psnr:+.6f} dSSIM={s-stop_ssim:+.6f} n={n}\n")

    print("[Saved]", summary_path)


if __name__ == "__main__":
    # Windows multiprocessing safety (especially if num_workers>0)
    import multiprocessing as mp
    try:
        mp.freeze_support()
    except Exception:
        pass
    main()
