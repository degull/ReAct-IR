# scripts/make_rollouts_valuehead.py
import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any, List, Optional, Set, Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# Make project import-safe (THIS FIXES "No module named models")
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ------------------------------
# Actions / scales
# ------------------------------
ACTIONS = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------------
# Basic utils
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def append_jsonl(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def to_torch(img: np.ndarray, device: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Always create float32 on CPU first, then move to device.
    If dtype is provided, cast to that dtype on the target device.
    """
    img = np.asarray(img, dtype=np.float32)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1x3xHxW
    x = x.to(device)
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x


def _unwrap_output(y: Any) -> torch.Tensor:
    """
    ToolBank/VETNet outputs can be:
      - Tensor
      - (Tensor, ...) / [Tensor, ...]
      - dict like {'output': Tensor} / {'pred': Tensor} / {'restored': Tensor} etc.
    We robustly pick the "main" tensor.
    """
    if torch.is_tensor(y):
        return y

    if isinstance(y, (list, tuple)) and len(y) > 0:
        # pick first tensor-like
        for v in y:
            if torch.is_tensor(v):
                return v
        # fallback: first element (may still be dict)
        return _unwrap_output(y[0])

    if isinstance(y, dict):
        # common keys
        for k in ["output", "pred", "restored", "y", "img", "out"]:
            if k in y and torch.is_tensor(y[k]):
                return y[k]
        # otherwise: first tensor value
        for _, v in y.items():
            if torch.is_tensor(v):
                return v
        # last resort
        raise TypeError(f"Dict output has no tensor values. keys={list(y.keys())}")

    raise TypeError(f"Unsupported output type: {type(y)}")


def to_numpy(y: torch.Tensor) -> np.ndarray:
    """
    Robustly convert model output to HWC float32 in [0,1].

    Accepts:
      - 1xCxHxW
      - BxCxHxW (uses first)
      - CxHxW
      - HxWxC
      - HxW (grayscale)
    """
    if not torch.is_tensor(y):
        raise TypeError(f"to_numpy expects Tensor, got: {type(y)}")

    y = y.detach()

    # Move/cast safely
    if y.is_cuda:
        y = y.float().cpu()
    else:
        y = y.float()

    # Normalize shape
    if y.ndim == 4:
        # BxCxHxW -> take first
        y = y[0]
    if y.ndim == 3:
        # could be CxHxW or HxWxC
        # Heuristic: if first dim is 1/3/4 it's probably CHW
        if y.shape[0] in (1, 3, 4) and (y.shape[1] > 4 and y.shape[2] > 4):
            y = y.permute(1, 2, 0)  # HWC
        else:
            # assume already HWC
            pass
    elif y.ndim == 2:
        # HxW -> add channel
        y = y.unsqueeze(-1)
    else:
        raise ValueError(f"Unexpected tensor ndim={y.ndim} shape={tuple(y.shape)}")

    y = y.clamp(0, 1).contiguous().numpy()

    # Ensure 3 channels RGB for metrics consistency
    if y.shape[-1] == 1:
        y = np.repeat(y, 3, axis=-1)
    elif y.shape[-1] == 4:
        # drop alpha
        y = y[..., :3]

    return y.astype(np.float32)


# ------------------------------
# PSNR / SSIM
# ------------------------------
def psnr_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.clip(pred, 0.0, 1.0).astype(np.float64)
    gt = np.clip(gt, 0.0, 1.0).astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse < eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _gaussian_kernel_1d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ax = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(ax * ax) / (2.0 * (sigma ** 2)))
    kernel = kernel / kernel.sum()
    return kernel


def ssim_torch(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    pred, gt: 1x3xHxW in [0,1]
    """
    if pred.dtype != gt.dtype:
        gt = gt.to(dtype=pred.dtype)
    if pred.device != gt.device:
        gt = gt.to(device=pred.device)

    device = pred.device
    dtype = pred.dtype
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    k1d = _gaussian_kernel_1d(window_size, sigma, device=device, dtype=dtype)
    k2d = torch.outer(k1d, k1d).unsqueeze(0).unsqueeze(0)  # 1x1xWxW

    def conv(img: torch.Tensor):
        ch = img.shape[1]
        weight = k2d.repeat(ch, 1, 1, 1)
        return F.conv2d(img, weight, padding=window_size // 2, groups=ch)

    mu1 = conv(pred)
    mu2 = conv(gt)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv(pred * pred) - mu1_sq
    sigma2_sq = conv(gt * gt) - mu2_sq
    sigma12 = conv(pred * gt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


# ------------------------------
# State feature (fallback)
# ------------------------------
def _fix_len(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == L:
        return x.astype(np.float32)
    y = np.zeros((L,), dtype=np.float32)
    n = min(L, x.shape[0])
    y[:n] = x[:n]
    return y.astype(np.float32)


def build_state_dict_from_arrays(s0: np.ndarray, m0_mean: np.ndarray, m0_max: np.ndarray) -> Dict[str, Any]:
    return {
        "s0": _fix_len(s0, 5).tolist(),
        "m0_mean": _fix_len(m0_mean, 5).tolist(),
        "m0_max": _fix_len(m0_max, 5).tolist(),
    }


class DiagnoserWrapper:
    def __init__(self, device: str):
        self.device = device
        print("[Diagnoser] Using fallback heuristic diagnoser (replace later with real diagnoser if needed).")

    @torch.no_grad()
    def __call__(self, x_img: np.ndarray) -> Dict[str, Any]:
        img = (x_img * 255.0).astype(np.uint8)
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32) / 255.0

        lap = (-4 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))
        var_lap = float(np.var(lap))
        blur_score = float(np.clip(1.0 - (var_lap * 10.0), 0.0, 1.0))

        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        haze_score = float(np.clip((brightness - 0.4) + (0.2 - contrast) * 2.0, 0.0, 1.0))

        dy = np.abs(gray - np.roll(gray, 1, 0))
        dx = np.abs(gray - np.roll(gray, 1, 1))
        rain_score = float(np.clip((dy.mean() - dx.mean()) * 10.0, 0.0, 1.0))

        snow_score = float(np.clip((gray > 0.85).mean() * 3.0, 0.0, 1.0))
        drop_score = float(np.clip((gray > 0.95).mean() * 6.0, 0.0, 1.0))

        s0 = np.asarray([blur_score, rain_score, snow_score, haze_score, drop_score], dtype=np.float32)
        m0_mean = s0 * 0.05
        m0_max = np.maximum(s0 * 0.2, s0.max() * 0.2)
        return build_state_dict_from_arrays(s0, m0_mean, m0_max)


# ------------------------------
# Project Restorer (VETNet + ToolBank)
# ------------------------------
def find_lora_ckpt(lora_dir: str, action: str) -> Optional[str]:
    if not lora_dir or not os.path.isdir(lora_dir):
        return None
    candidates = [
        os.path.join(lora_dir, action.replace("A_", "").lower()),
        os.path.join(lora_dir, action),
        os.path.join(lora_dir, action.lower()),
        lora_dir,
    ]
    best = None
    best_mtime = -1.0
    for c in candidates:
        if os.path.isdir(c):
            for fn in os.listdir(c):
                if fn.lower().endswith(".pth"):
                    fp = os.path.join(c, fn)
                    mt = os.path.getmtime(fp)
                    if mt > best_mtime:
                        best_mtime = mt
                        best = fp
        elif os.path.isfile(c) and c.lower().endswith(".pth"):
            mt = os.path.getmtime(c)
            if mt > best_mtime:
                best_mtime = mt
                best = c
    return best


def extract_lora_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "lora_state_dict" in ckpt_obj and isinstance(ckpt_obj["lora_state_dict"], dict):
            return ckpt_obj["lora_state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        return ckpt_obj
    raise TypeError(f"Unsupported ckpt type: {type(ckpt_obj)}")


class RestorerWrapper:
    """
    Wrapper for ReAct-IR ToolBank API.
    """
    def __init__(
        self,
        device: str,
        backbone_ckpt: str,
        lora_dir: str,
        use_amp: int = 1,
        rank: int = 2,
        alpha: float = 1.0,
        wrap_only_1x1: int = 1,
        strict_lora: int = 0,
        dim: int = 64,
        bias: int = 1,
        volterra_rank: int = 2,
    ):
        self.device = device
        self.use_amp = int(use_amp) == 1

        from models.backbone.vetnet import VETNet  # noqa
        from models.toolbank.toolbank import ToolBank  # noqa

        backbone = VETNet(dim=int(dim), bias=bool(int(bias)), volterra_rank=int(volterra_rank)).to(device).eval()

        # ------------------------------
        # FIX: load correct state_dict + strip prefixes
        # ------------------------------
        def strip_prefix(state: Dict[str, torch.Tensor], pref: str) -> Dict[str, torch.Tensor]:
            if any(k.startswith(pref) for k in state.keys()):
                return {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
            return state

        if backbone_ckpt and os.path.isfile(backbone_ckpt):
            ckpt = torch.load(backbone_ckpt, map_location="cpu")

            # 1) prefer ckpt["state_dict"] if present
            sd = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict)) else ckpt

            # 2) strip common prefixes
            sd = strip_prefix(sd, "module.")
            sd = strip_prefix(sd, "backbone.")

            ret = backbone.load_state_dict(sd, strict=False)
            print(f"[Backbone] loaded: {backbone_ckpt} (strict=False)")
            print(f"[Backbone] missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}")
            if len(ret.missing_keys) > 0:
                print("[Backbone] WARNING: missing keys suggests backbone config (dim/bias/volterra_rank) mismatch OR wrong key-prefix.")
        else:
            print(f"[Backbone] WARNING: backbone_ckpt not found: {backbone_ckpt}")

        self.toolbank = ToolBank(
            backbone=backbone,
            actions=ACTIONS,
            rank=int(rank),
            alpha=float(alpha),
            wrap_only_1x1=bool(int(wrap_only_1x1)),
        ).to(device).eval()

        found = 0
        for act in ACTIONS:
            ckpt_path = find_lora_ckpt(lora_dir, act)
            if ckpt_path is None:
                print(f"[LoRA] WARNING: not found for {act} under {lora_dir} -> keep as init")
                continue

            raw = torch.load(ckpt_path, map_location="cpu")
            lora_sd = extract_lora_state_dict(raw)

            if found == 0:
                print("[LoRA] sample keys:", list(lora_sd.keys())[:5])

            info = self.toolbank.load_lora_state_dict_for_action(
                action=act,
                state_dict=lora_sd,
                strict=bool(int(strict_lora)),
                map_location="cpu",
            )
            found += 1
            print(
                f"[LoRA] loaded {act} <- {ckpt_path} | "
                f"expected={info['expected_lora_keys']} provided={info['provided_keys']} missing_expected={info['missing_expected']}"
            )

        print(f"[LoRA] loaded_actions={found}/{len(ACTIONS)} from {lora_dir}")

        self.toolbank.activate("A_STOP", 0.0)

    @torch.no_grad()
    def apply(self, x_img: np.ndarray, action: str, scale: float) -> np.ndarray:
        xt = to_torch(x_img, self.device)

        self.toolbank.activate(action, float(scale))

        amp_enabled = (self.use_amp and self.device.startswith("cuda"))
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            y = self.toolbank(xt)

        yt = _unwrap_output(y)
        return to_numpy(yt)


# ------------------------------
# Pair collectors
# ------------------------------
def list_files_recursive(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(dp, fn))
    return out


def collect_pairs_day_raindrop(data_root: str) -> List[Dict[str, Any]]:
    base = os.path.join(data_root, "DayRainDrop")
    drop_root = os.path.join(base, "Drop")
    clear_root = os.path.join(base, "Clear")
    if not (os.path.isdir(drop_root) and os.path.isdir(clear_root)):
        return []
    inputs = list_files_recursive(drop_root)
    pairs = []
    for ip in inputs:
        rel = os.path.relpath(ip, drop_root).replace("\\", "/")
        gp = os.path.join(clear_root, rel)
        if os.path.isfile(gp):
            key = os.path.splitext(rel)[0]
            pairs.append(
                {"input": ip.replace("\\", "/"), "gt": gp.replace("\\", "/"),
                 "dataset": "DayRainDrop", "split": "train", "key": key, "source_action": "A_DEDROP"}
            )
    return pairs


def collect_pairs_night_raindrop(data_root: str) -> List[Dict[str, Any]]:
    base = os.path.join(data_root, "NightRainDrop")
    drop_root = os.path.join(base, "Drop")
    clear_root = os.path.join(base, "Clear")
    if not (os.path.isdir(drop_root) and os.path.isdir(clear_root)):
        return []
    inputs = list_files_recursive(drop_root)
    pairs = []
    for ip in inputs:
        rel = os.path.relpath(ip, drop_root).replace("\\", "/")
        gp = os.path.join(clear_root, rel)
        if os.path.isfile(gp):
            key = os.path.splitext(rel)[0]
            pairs.append(
                {"input": ip.replace("\\", "/"), "gt": gp.replace("\\", "/"),
                 "dataset": "NightRainDrop", "split": "train", "key": key, "source_action": "A_DEDROP"}
            )
    return pairs


def collect_pairs_reside6k(data_root: str) -> List[Dict[str, Any]]:
    base = os.path.join(data_root, "RESIDE-6K", "train")
    hazy_root = os.path.join(base, "hazy")
    gt_root = os.path.join(base, "GT")
    if not (os.path.isdir(hazy_root) and os.path.isdir(gt_root)):
        return []
    inputs = list_files_recursive(hazy_root)
    pairs = []
    for ip in inputs:
        fn = os.path.basename(ip)
        gp = os.path.join(gt_root, fn)
        if os.path.isfile(gp):
            key = os.path.splitext(fn)[0]
            pairs.append(
                {"input": ip.replace("\\", "/"), "gt": gp.replace("\\", "/"),
                 "dataset": "RESIDE-6K", "split": "train", "key": key, "source_action": "A_DEHAZE"}
            )
    return pairs


def collect_pairs_csd(data_root: str) -> List[Dict[str, Any]]:
    base = os.path.join(data_root, "CSD", "Train")
    snow_root = os.path.join(base, "Snow")
    gt_root = os.path.join(base, "Gt")
    if not (os.path.isdir(snow_root) and os.path.isdir(gt_root)):
        return []
    inputs = list_files_recursive(snow_root)
    pairs = []
    for ip in inputs:
        fn = os.path.basename(ip)
        gp = os.path.join(gt_root, fn)
        if os.path.isfile(gp):
            key = os.path.splitext(fn)[0]
            pairs.append(
                {"input": ip.replace("\\", "/"), "gt": gp.replace("\\", "/"),
                 "dataset": "CSD", "split": "train", "key": key, "source_action": "A_DESNOW"}
            )
    return pairs


def collect_pairs_rain100h(data_root: str) -> List[Dict[str, Any]]:
    base = os.path.join(data_root, "rain100H", "train")
    rain_root = os.path.join(base, "rain")
    gt_root = os.path.join(base, "norain")
    if not (os.path.isdir(rain_root) and os.path.isdir(gt_root)):
        return []
    inputs = list_files_recursive(rain_root)
    pairs = []
    for ip in inputs:
        fn = os.path.basename(ip)
        gp = os.path.join(gt_root, fn)
        if os.path.isfile(gp):
            key = os.path.splitext(fn)[0]
            pairs.append(
                {"input": ip.replace("\\", "/"), "gt": gp.replace("\\", "/"),
                 "dataset": "rain100H", "split": "train", "key": key, "source_action": "A_DERAIN"}
            )
    return pairs


def collect_all_pairs(data_root: str) -> List[Dict[str, Any]]:
    pairs = []
    pairs += collect_pairs_day_raindrop(data_root)
    pairs += collect_pairs_night_raindrop(data_root)
    pairs += collect_pairs_reside6k(data_root)
    pairs += collect_pairs_csd(data_root)
    pairs += collect_pairs_rain100h(data_root)
    return pairs


def choose_samples_balanced(pairs: List[Dict[str, Any]], target_items: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_ds.setdefault(p["dataset"], []).append(p)
    for ds in by_ds:
        rng.shuffle(by_ds[ds])
    datasets = list(by_ds.keys())
    rng.shuffle(datasets)

    out = []
    idx = {ds: 0 for ds in datasets}
    while len(out) < target_items:
        progressed = False
        for ds in datasets:
            lst = by_ds[ds]
            if idx[ds] < len(lst):
                out.append(lst[idx[ds]])
                idx[ds] += 1
                progressed = True
                if len(out) >= target_items:
                    break
        if not progressed:
            break
    return out


def parse_scales(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if 0.0 not in out:
        out = [0.0] + out
    return out


def make_meta(pair: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": pair.get("source_action", "A_DEBLUR"),
        "split": pair.get("split", "train"),
        "dataset": pair.get("dataset", "Unknown"),
        "key": pair.get("key", ""),
        "input_path": pair["input"],
        "gt_path": pair["gt"],
        "source_action": pair.get("source_action", "A_DEBLUR"),
    }


def load_existing_keys(out_jsonl: str) -> Set[str]:
    keys: Set[str] = set()
    if not out_jsonl or not os.path.isfile(out_jsonl):
        return keys
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
                meta = it.get("meta", {})
                ds = meta.get("dataset", "")
                k = meta.get("key", "")
                if ds and k:
                    keys.add(f"{ds}:{k}")
            except Exception:
                continue
    return keys


def choose_best(sweep: List[Dict[str, Any]], ref_psnr: float, ref_ssim: float, w_ssim: float) -> Dict[str, Any]:
    best = None
    best_score = -1e18
    for e in sweep:
        dps = float(e["psnr"] - ref_psnr)
        dss = float(e["ssim"] - ref_ssim)
        score = dps + w_ssim * dss
        if score > best_score:
            best_score = score
            best = {
                "action": e["action"],
                "scale": float(e["scale"]),
                "psnr": float(e["psnr"]),
                "ssim": float(e["ssim"]),
                "d_psnr": float(dps),
                "d_ssim": float(dss),
            }
    return best if best is not None else {
        "action": "A_DEBLUR",
        "scale": 0.0,
        "psnr": float(ref_psnr),
        "ssim": float(ref_ssim),
        "d_psnr": 0.0,
        "d_ssim": 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="E:/ReAct-IR/data")
    ap.add_argument("--target_items", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--backbone_ckpt", type=str, required=True)
    ap.add_argument("--lora_dir", type=str, required=True)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--batch_gpu", type=int, default=1)

    # backbone config
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--volterra_rank", type=int, default=2)

    ap.add_argument("--lora_rank", type=int, default=2)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--wrap_only_1x1", type=int, default=1)
    ap.add_argument("--strict_lora", type=int, default=0)

    ap.add_argument("--scales", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--scalar_w_ssim", type=float, default=100.0)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    scales = parse_scales(args.scales)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Out] {args.out_jsonl}")
    print(f"[DataRoot] {args.data_root}")
    print(f"[Target] items={args.target_items} scales={scales}")
    print(f"[BackboneCfg] dim={args.dim} bias={args.bias} volterra_rank={args.volterra_rank}")
    print(f"[LoRACfg] rank={args.lora_rank} alpha={args.lora_alpha} wrap_only_1x1={args.wrap_only_1x1} strict={args.strict_lora}")

    all_pairs = collect_all_pairs(args.data_root)
    print(f"[Collect] total_pairs={len(all_pairs)} from datasets={sorted(list(set([p['dataset'] for p in all_pairs])))}")
    if len(all_pairs) == 0:
        raise RuntimeError("No dataset pairs found. Check --data_root folder structure.")

    existing = load_existing_keys(args.out_jsonl)
    if existing:
        print(f"[Resume] existing_keys={len(existing)} (will skip duplicates)")

    chosen = choose_samples_balanced(all_pairs, args.target_items + len(existing), args.seed)
    print(f"[Choose] chosen_pairs={len(chosen)} (balanced round-robin)")

    diagnoser = DiagnoserWrapper(device=device)
    restorer = RestorerWrapper(
        device=device,
        backbone_ckpt=args.backbone_ckpt,
        lora_dir=args.lora_dir,
        use_amp=args.use_amp,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        wrap_only_1x1=args.wrap_only_1x1,
        strict_lora=args.strict_lora,
        dim=args.dim,
        bias=args.bias,
        volterra_rank=args.volterra_rank,
    )

    # id continuation
    next_id = 1
    if os.path.isfile(args.out_jsonl):
        try:
            with open(args.out_jsonl, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                pos = max(0, end - 4096)
                f.seek(pos)
                tail = f.read().decode("utf-8", errors="ignore").splitlines()
                for ln in reversed(tail):
                    ln = ln.strip()
                    if ln:
                        next_id = int(json.loads(ln).get("id", 0)) + 1
                        break
        except Exception:
            pass

    t0 = time.time()
    made = 0
    processed = 0

    for pair in chosen:
        ds = pair["dataset"]
        key = pair["key"]
        uniq = f"{ds}:{key}"
        if uniq in existing:
            processed += 1
            continue

        ip = pair["input"]
        gp = pair["gt"]
        if not (os.path.isfile(ip) and os.path.isfile(gp)):
            processed += 1
            continue

        try:
            x_img = read_image_rgb(ip)
            gt_img = read_image_rgb(gp)
        except Exception as e:
            print(f"[Skip] read error: {ip} ({repr(e)})")
            processed += 1
            continue

        state = diagnoser(x_img)

        # metrics baseline (input)
        x_t = to_torch(x_img, device)
        gt_t = to_torch(gt_img, device)
        psnr_in = psnr_np(x_img, gt_img)
        ssim_in = ssim_torch(x_t, gt_t)

        # base (A_STOP)
        base_action = "A_STOP"
        base_scale = 0.0
        base_img = restorer.apply(x_img, base_action, base_scale)

        psnr_base = psnr_np(base_img, gt_img)
        base_t = to_torch(base_img, device)
        ssim_base = ssim_torch(base_t, gt_t)

        baseline = {
            "psnr_in": float(psnr_in),
            "ssim_in": float(ssim_in),
            "psnr_base": float(psnr_base),
            "ssim_base": float(ssim_base),
            "base_scale": float(base_scale),
            "base_from_action": base_action,
        }

        # sweep
        sweep = []
        for act in ACTIONS:
            for sc in scales:
                out_img = restorer.apply(x_img, act, float(sc))
                p = psnr_np(out_img, gt_img)
                out_t = to_torch(out_img, device)
                s = ssim_torch(out_t, gt_t)

                sweep.append({
                    "action": act,
                    "scale": float(sc),
                    "psnr": float(p),
                    "ssim": float(s),
                    "d_psnr": float(p - psnr_base),
                    "d_ssim": float(s - ssim_base),
                })

        best = choose_best(sweep, ref_psnr=psnr_base, ref_ssim=ssim_base, w_ssim=float(args.scalar_w_ssim))
        best_old = choose_best(sweep, ref_psnr=psnr_in, ref_ssim=ssim_in, w_ssim=float(args.scalar_w_ssim))

        item = {
            "id": int(next_id),
            "input": ip,
            "gt": gp,
            "meta": make_meta(pair),
            "state": state,
            "baseline": baseline,
            "sweep": sweep,
            "best": best,
            "best_old": best_old,
        }
        append_jsonl(args.out_jsonl, item)

        existing.add(uniq)
        next_id += 1
        made += 1
        processed += 1

        if made % args.log_every == 0:
            dt = time.time() - t0
            rate = made / max(dt, 1e-6)
            eta = (args.target_items - made) / max(rate, 1e-6) if made < args.target_items else 0.0
            print(
                f"[Prog] made={made} processed={processed} rate={rate:.2f} it/s eta={eta/60:.1f} min | "
                f"last={ds}:{key} best={best['action']}@{best['scale']} dP={best['d_psnr']:.3f} dS={best['d_ssim']:.4f}"
            )

        if made >= args.target_items:
            break

    print(f"[Done] made={made} (target={args.target_items}) out={args.out_jsonl}")


if __name__ == "__main__":
    main()
