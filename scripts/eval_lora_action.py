# scripts/eval_lora_action.py
# ------------------------------------------------------------
# Standalone LoRA-tool evaluator (NO dependency on datasets/pairs.py)
#
# + Cached auto-pick backbone:
#   - If --backbone_ckpt is empty, scan --backbone_dir/*.pth
#   - infer dim from state_dict["patch_embed.weight"].shape[0]
#   - filter by --backbone_dim_target
#   - choose by --backbone_pick (epoch|mtime)
#   - cache results to: backbone_dir/_ckpt_dim_cache.json
#     (next runs only scan new/modified ckpts)
#
# + Sanity check (pairing + TRUE backbone-only + toolbank pred) on first iter:
#   - PSNR/SSIM(input, gt), meta0
#   - PSNR/SSIM(backbone(scale=0), gt)   <-- IMPORTANT: force LoRA off
#   - PSNR/SSIM(pred(scale=cfg.scale), gt)
#   - prints tensor range/shape for inp/bb/pred
#   - prints |pred-bb| mean/max to verify LoRA effect
#
# + LoRA load patch:
#   - strip common prefixes
#   - auto add "backbone." prefix if needed
#   - try strict=True then fallback strict=False with diagnostics
#
# Datasets expected under --data_root:
#   CSD/{Train,Test}/{Snow,Gt}
#   rain100H/{train,test}/{rain,norain}
#   RESIDE-6K/{train,test}/{hazy,GT}
#   DayRainDrop/{Drop,Clear,Blur}
#   NightRainDrop/{Drop,Clear}
# ------------------------------------------------------------

# scripts/eval_lora_action.py
# ------------------------------------------------------------
# Standalone LoRA-tool evaluator (NO dependency on datasets/pairs.py)
#
# + Cached auto-pick backbone:
#   - If --backbone_ckpt is empty, scan --backbone_dir/*.pth
#   - infer dim from state_dict["patch_embed.weight"].shape[0]
#   - filter by --backbone_dim_target
#   - choose by --backbone_pick (epoch|mtime)
#   - cache results to: backbone_dir/_ckpt_dim_cache.json
#
# + Sanity check (pairing + bb(scale=0) + pred(toolbank@scale)) on first iter of EACH sweep:
#   - PSNR/SSIM(input, gt), meta0
#   - PSNR/SSIM(backbone(scale=0), gt)
#   - PSNR/SSIM(pred(scale=s), gt)
#   - |pred-bb| mean/max
#   - prints tensor range/shape for inp/bb/pred
#
# + LoRA load patch:
#   - strip common prefixes
#   - auto add "backbone." prefix if needed
#   - try strict=True then fallback strict=False with diagnostics
#
# + NEW: --eval_scales "0,1" (scale sweep in ONE run)
#   - runs full dataset per scale
#   - writes outputs under results_root/<action>/<split>_<split_mode>/scale_<s>/
#   - prints summary table + ΔPSNR/ΔSSIM vs the first scale
# ------------------------------------------------------------

""" import os
import sys
import glob
import re
import time
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm

# ------------------------------------------------------------
# Make project import-safe
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ------------------------------------------------------------
# Optional skimage metrics
# ------------------------------------------------------------
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False


# ============================================================
# Utils
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec - 60 * m)
    h = int(m // 60)
    m = int(m - 60 * h)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def is_img(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def list_images(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    files = []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            p = os.path.join(dp, f)
            if is_img(p):
                files.append(p)
    files.sort()
    return files


def make_key_from_path(p: str, root: str) -> str:
    rp = os.path.relpath(p, root).replace("\\", "/")
    rp = os.path.splitext(rp)[0]
    return rp


def imread_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_chw_float01(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1)  # CHW


def tensor_to_img_u8(t_chw: torch.Tensor) -> np.ndarray:
    t = t_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255.0).round().astype(np.uint8)


def save_img_u8(img_u8: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8).save(path)


def save_triplet(inp_chw: torch.Tensor, pred_chw: torch.Tensor, gt_chw: torch.Tensor, path: str):
    inp = tensor_to_img_u8(inp_chw)
    pr = tensor_to_img_u8(pred_chw)
    gt = tensor_to_img_u8(gt_chw)
    h, w, _ = inp.shape
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = inp
    canvas[:, w:2 * w] = pr
    canvas[:, 2 * w:3 * w] = gt
    save_img_u8(canvas, path)


def compute_psnr_ssim(pred_u8: np.ndarray, gt_u8: np.ndarray) -> Tuple[float, float]:
    if not USE_SKIMAGE:
        return 0.0, 0.0
    ps = float(peak_signal_noise_ratio(gt_u8, pred_u8, data_range=255))
    ss = float(structural_similarity(gt_u8, pred_u8, channel_axis=2, data_range=255))
    return ps, ss


def pad_to_multiple_of(x: torch.Tensor, mult: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    pad = (0, pad_w, 0, pad_h)  # (L,R,T,B)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, pad, mode="reflect")
    return x, pad


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    l, r, t, b = pad
    if r == 0 and b == 0 and l == 0 and t == 0:
        return x
    h = x.shape[-2]
    w = x.shape[-1]
    return x[..., t : h - b, l : w - r]


def _tensor_range(x: torch.Tensor) -> Tuple[float, float]:
    x = x.detach()
    return float(x.min().item()), float(x.max().item())


def _safe_scale_name(s: float) -> str:
    # 0.5 -> "0p5", 1.0 -> "1", -0.25 -> "m0p25"
    sign = "m" if s < 0 else ""
    s = abs(float(s))
    if abs(s - int(s)) < 1e-9:
        return f"{sign}{int(s)}"
    st = f"{s:.6f}".rstrip("0").rstrip(".")
    return sign + st.replace(".", "p")


def parse_eval_scales(eval_scales: str, fallback_single: float) -> List[float]:
    if eval_scales is None or str(eval_scales).strip() == "":
        return [float(fallback_single)]
    parts = [p.strip() for p in str(eval_scales).split(",") if p.strip() != ""]
    if not parts:
        return [float(fallback_single)]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            raise ValueError(f"Invalid --eval_scales item: '{p}' (expected comma-separated floats like '0,1,0.5')")
    return out


# ============================================================
# Action mapping
# ============================================================
ALIAS_TO_INTERNAL = {
    "dedrop": "A_DEDROP",
    "desnow": "A_DESNOW",
    "derain": "A_DERAIN",
    "deblur": "A_DEBLUR",
    "dehaze": "A_DEHAZE",
}
INTERNAL_ACTIONS = ["A_DEDROP", "A_DESNOW", "A_DERAIN", "A_DEBLUR", "A_DEHAZE"]


# ============================================================
# Pair scanners (standalone)
# ============================================================
def scan_pairs_by_stem(input_root: str, gt_root: str, name: str) -> Tuple[List[Tuple[str, str, Dict]], Dict]:
    in_files = list_images(input_root)
    gt_files = list_images(gt_root)

    in_map: Dict[str, str] = {}
    gt_map: Dict[str, str] = {}

    for p in in_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in in_map:
            in_map[stem] = p

    for p in gt_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in gt_map:
            gt_map[stem] = p

    keys = sorted(set(in_map.keys()) & set(gt_map.keys()))
    pairs: List[Tuple[str, str, Dict]] = []
    for k in keys:
        pairs.append((in_map[k], gt_map[k], {"key": f"{name}_{k}", "stem": k}))

    report = {
        "name": name,
        "input_root": input_root,
        "gt_root": gt_root,
        "input_total": len(in_files),
        "gt_total": len(gt_files),
        "paired": len(pairs),
        "missing_gt": len(set(in_map.keys()) - set(gt_map.keys())),
        "missing_input": len(set(gt_map.keys()) - set(in_map.keys())),
        "dup_input_stems": max(0, len(in_files) - len(in_map)),
        "dup_gt_stems": max(0, len(gt_files) - len(gt_map)),
    }
    return pairs, report


def scan_pairs_by_relpath(input_root: str, gt_root: str, name: str) -> Tuple[List[Tuple[str, str, Dict]], Dict]:
    in_files = list_images(input_root)
    gt_files = list_images(gt_root)

    in_map: Dict[str, str] = {}
    gt_map: Dict[str, str] = {}

    for p in in_files:
        k = make_key_from_path(p, input_root)
        if k not in in_map:
            in_map[k] = p

    for p in gt_files:
        k = make_key_from_path(p, gt_root)
        if k not in gt_map:
            gt_map[k] = p

    keys = sorted(set(in_map.keys()) & set(gt_map.keys()))
    pairs: List[Tuple[str, str, Dict]] = []
    for k in keys:
        pairs.append((in_map[k], gt_map[k], {"key": f"{name}_{k}", "rel": k}))

    report = {
        "name": name,
        "input_root": input_root,
        "gt_root": gt_root,
        "input_total": len(in_files),
        "gt_total": len(gt_files),
        "paired": len(pairs),
        "missing_gt": len(set(in_map.keys()) - set(gt_map.keys())),
        "missing_input": len(set(gt_map.keys()) - set(in_map.keys())),
        "dup_input_keys": max(0, len(in_files) - len(in_map)),
        "dup_gt_keys": max(0, len(gt_files) - len(gt_map)),
    }
    return pairs, report


def build_pairs_for_action(data_root: str, action_internal: str, split: str) -> Tuple[List[Tuple[str, str, Dict]], List[Dict]]:
    data_root = data_root.replace("\\", "/")
    split = split.lower().strip()

    reports: List[Dict] = []
    pairs: List[Tuple[str, str, Dict]] = []

    if action_internal == "A_DESNOW":
        if split not in ["train", "test"]:
            raise ValueError("A_DESNOW split must be train or test")
        ip = os.path.join(data_root, "CSD", split.capitalize(), "Snow")
        gp = os.path.join(data_root, "CSD", split.capitalize(), "Gt")
        pr, rep = scan_pairs_by_stem(ip, gp, name=f"csd_{split}")
        pairs.extend(pr)
        reports.append(rep)

    elif action_internal == "A_DERAIN":
        if split not in ["train", "test"]:
            raise ValueError("A_DERAIN split must be train or test")
        ip = os.path.join(data_root, "rain100H", split, "rain")
        gp = os.path.join(data_root, "rain100H", split, "norain")
        pr, rep = scan_pairs_by_stem(ip, gp, name=f"rain100H_{split}")
        pairs.extend(pr)
        reports.append(rep)

    elif action_internal == "A_DEHAZE":
        if split not in ["train", "test"]:
            raise ValueError("A_DEHAZE split must be train or test")
        ip = os.path.join(data_root, "RESIDE-6K", split, "hazy")
        gp = os.path.join(data_root, "RESIDE-6K", split, "GT")
        pr, rep = scan_pairs_by_stem(ip, gp, name=f"reside6k_{split}")
        pairs.extend(pr)
        reports.append(rep)

    elif action_internal == "A_DEDROP":
        day_in = os.path.join(data_root, "DayRainDrop", "Drop")
        day_gt = os.path.join(data_root, "DayRainDrop", "Clear")
        pr1, rep1 = scan_pairs_by_relpath(day_in, day_gt, name="dedrop_day")
        if len(pr1) == 0:
            pr1, rep1 = scan_pairs_by_stem(day_in, day_gt, name="dedrop_day_stem")
        pairs.extend(pr1)
        reports.append(rep1)

        night_in = os.path.join(data_root, "NightRainDrop", "Drop")
        night_gt = os.path.join(data_root, "NightRainDrop", "Clear")
        pr2, rep2 = scan_pairs_by_relpath(night_in, night_gt, name="dedrop_night")
        if len(pr2) == 0:
            pr2, rep2 = scan_pairs_by_stem(night_in, night_gt, name="dedrop_night_stem")
        pairs.extend(pr2)
        reports.append(rep2)

    elif action_internal == "A_DEBLUR":
        blur_in = os.path.join(data_root, "DayRainDrop", "Blur")
        clear_gt = os.path.join(data_root, "DayRainDrop", "Clear")
        pr, rep = scan_pairs_by_relpath(blur_in, clear_gt, name="deblur_day")
        if len(pr) == 0:
            pr, rep = scan_pairs_by_stem(blur_in, clear_gt, name="deblur_day_stem")
        pairs.extend(pr)
        reports.append(rep)

    else:
        raise ValueError(f"Unknown action_internal: {action_internal}")

    return pairs, reports


# ============================================================
# Dataset
# ============================================================
class PairedEvalDataset(Dataset):
    def __init__(self, pairs, limit=0, seed=123, val_ratio=0.0, split_mode="all"):
        self.all_pairs = list(pairs)

        if limit and limit > 0:
            rng = np.random.RandomState(seed)
            idxs = rng.permutation(len(self.all_pairs))[: int(limit)]
            self.all_pairs = [self.all_pairs[i] for i in idxs]

        self.pairs = self.all_pairs
        val_ratio = float(val_ratio)
        if val_ratio > 0.0:
            rng = np.random.RandomState(seed)
            n = len(self.all_pairs)
            idxs = rng.permutation(n)
            n_val = max(1, int(round(n * val_ratio)))
            val_set = set(idxs[:n_val].tolist())
            if split_mode == "val":
                self.pairs = [p for i, p in enumerate(self.all_pairs) if i in val_set]
            else:
                self.pairs = [p for i, p in enumerate(self.all_pairs) if i not in val_set]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, gt_path, meta = self.pairs[idx]
        inp = to_chw_float01(imread_rgb(inp_path))
        gt = to_chw_float01(imread_rgb(gt_path))
        if "key" not in meta:
            meta = dict(meta)
            meta["key"] = os.path.splitext(os.path.basename(inp_path))[0]
        return {"input": inp, "gt": gt, "meta": meta}


def collate_eval(batch):
    xs = torch.stack([b["input"] for b in batch], dim=0)
    ys = torch.stack([b["gt"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return xs, ys, metas


# ============================================================
# CKPT helpers + Auto-pick backbone (CACHED)
# ============================================================
def safe_load_state_dict_from_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported ckpt format: {ckpt_path}")


def safe_load_cfg_from_ckpt(ckpt_path: str) -> Dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "cfg" in obj and isinstance(obj["cfg"], dict):
        return obj["cfg"]
    return {}


def infer_backbone_dim_from_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    w = sd.get("patch_embed.weight", None)
    if w is None:
        return None
    return int(w.shape[0])


def parse_epoch_from_name(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"epoch[_\-]?(\d+)", base, flags=re.IGNORECASE)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _cache_path(backbone_dir: str) -> str:
    return os.path.join(backbone_dir, "_ckpt_dim_cache.json")


def _load_cache(backbone_dir: str) -> Dict[str, Dict]:
    cp = _cache_path(backbone_dir)
    if os.path.isfile(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(backbone_dir: str, cache: Dict[str, Dict]):
    cp = _cache_path(backbone_dir)
    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def pick_backbone_ckpt(
    backbone_dir: str,
    target_dim: int,
    pick_mode: str = "epoch",  # "epoch" | "mtime"
    verbose: bool = True,
    max_scan: int = 0,  # 0 = no limit
) -> str:

    backbone_dir = backbone_dir.replace("\\", "/")
    if not os.path.isdir(backbone_dir):
        raise RuntimeError(f"--backbone_dir not found: {backbone_dir}")

    cands = sorted(glob.glob(os.path.join(backbone_dir, "*.pth")))
    if not cands:
        raise RuntimeError(f"No .pth files in backbone_dir: {backbone_dir}")

    cache = _load_cache(backbone_dir)

    to_scan = []
    for p in cands:
        mt = os.path.getmtime(p)
        rec = cache.get(p)
        if rec is None or float(rec.get("mtime", -1)) != float(mt):
            to_scan.append(p)

    if max_scan and max_scan > 0:
        to_scan = to_scan[:max_scan]

    if verbose:
        print(f"[AutoPick] cache_entries={len(cache)} total_ckpts={len(cands)} need_scan={len(to_scan)} target_dim={target_dim}")

    if to_scan:
        for p in tqdm(to_scan, ncols=120, desc="AutoPick scan ckpts"):
            try:
                sd = safe_load_state_dict_from_ckpt(p)
                dim = infer_backbone_dim_from_sd(sd)
                ep = parse_epoch_from_name(p)
                mt = os.path.getmtime(p)
                cache[p] = {"dim": dim, "epoch": ep, "mtime": mt}
            except Exception as e:
                cache[p] = {
                    "dim": None,
                    "epoch": parse_epoch_from_name(p),
                    "mtime": os.path.getmtime(p),
                    "error": repr(e),
                }
        _save_cache(backbone_dir, cache)

    ok = []
    bad = 0
    for p in cands:
        rec = cache.get(p, {})
        dim = rec.get("dim", None)
        if dim is None:
            bad += 1
            continue
        if int(dim) != int(target_dim):
            continue
        ok.append((p, int(dim), int(rec.get("epoch", -1)), float(rec.get("mtime", 0.0))))

    if verbose:
        print(f"[AutoPick] matched={len(ok)} failed={bad}")

    if not ok:
        raise RuntimeError(f"No backbone ckpt with dim={target_dim} in {backbone_dir}")

    if pick_mode == "mtime":
        ok.sort(key=lambda x: x[3], reverse=True)
    else:
        ok.sort(key=lambda x: (x[2], x[3]), reverse=True)

    chosen = ok[0][0]
    if verbose:
        print("[AutoPick] chosen:", chosen)
    return chosen


# ============================================================
# Prefix helpers (strip)
# ============================================================
def _strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    if not isinstance(sd, dict) or not sd:
        return sd

    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if all(k.startswith(prefix) for k in d.keys()):
            return {k[len(prefix) :]: v for k, v in d.items()}
        return d

    sd2 = sd
    for p in ["module.", "model.", "state_dict."]:
        sd2 = strip_prefix(sd2, p)
    return sd2


def _maybe_add_backbone_prefix_for_lora(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    keys = list(sd.keys())
    has_backbone = any(k.startswith("backbone.") for k in keys)
    if has_backbone:
        return sd
    has_lora = any(".lora_A." in k or ".lora_B." in k for k in keys)
    if not has_lora:
        return sd
    return {("backbone." + k): v for k, v in sd.items()}


# ============================================================
# LoRA loading (patched)
# ============================================================
def load_lora_into_toolbank(tb: ToolBank, action_internal: str, lora_ckpt_path: str):
    obj = torch.load(lora_ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and ("lora_state_dict" in obj):
        sd = obj["lora_state_dict"]
    elif isinstance(obj, dict) and ("state_dict" in obj) and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    else:
        sd = obj

    sd = _strip_known_prefixes(sd)
    sd = _maybe_add_backbone_prefix_for_lora(sd)

    keys = list(sd.keys())
    n_keys = len(keys)
    n_lora = sum(1 for k in keys if (".lora_A." in k or ".lora_B." in k))
    print(f"[LoRA] sd keys={n_keys}  lora_keys={n_lora}")
    if n_keys > 0:
        print("[LoRA] key sample:", keys[:5])

    try:
        if hasattr(tb, "load_lora_state_dict_for_action"):
            info = tb.load_lora_state_dict_for_action(action_internal, sd, strict=True)
            print("[LoRA] load_lora_state_dict_for_action info:", info)
        elif hasattr(tb, "load_lora_state_dict"):
            info = tb.load_lora_state_dict(action_internal, sd, strict=True)
            print("[LoRA] load_lora_state_dict info:", info)
        else:
            raise AttributeError("ToolBank must provide load_lora_state_dict_for_action(...) or load_lora_state_dict(...)")
    except RuntimeError as e:
        msg = str(e)
        print("[LoRA] strict=True failed:", msg[:500], "..." if len(msg) > 500 else "")
        print("[LoRA] retry with strict=False (partial load) ...")
        if hasattr(tb, "load_lora_state_dict_for_action"):
            info = tb.load_lora_state_dict_for_action(action_internal, sd, strict=False)
            print("[LoRA] load_lora_state_dict_for_action info (strict=False):", info)
        else:
            info = tb.load_lora_state_dict(action_internal, sd, strict=False)
            print("[LoRA] load_lora_state_dict info (strict=False):", info)


def find_latest_ckpt_in_dir(action_alias: str, save_root: str) -> Optional[str]:
    d = os.path.join(save_root, action_alias)
    if not os.path.isdir(d):
        return None
    cands = glob.glob(os.path.join(d, "*.pth"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


# ============================================================
# Config
# ============================================================
@dataclass
class EvalConfig:
    action_alias: str
    action_internal: str

    backbone_ckpt: str
    backbone_dir: str
    backbone_dim_target: int
    backbone_pick: str
    backbone_max_scan: int

    lora_ckpt: str
    data_root: str
    results_root: str

    split: str
    split_mode: str
    val_ratio: float
    limit: int
    seed: int

    # single-scale mode (legacy)
    scale: float
    # sweep mode
    eval_scales: str
    no_sweep_save: bool

    batch_size: int
    num_workers: int
    use_amp: bool
    channels_last: bool
    tf32: bool

    dim: int
    bias: bool
    volterra_rank: int

    save_images: bool
    save_triplets: bool


def parse_args() -> EvalConfig:
    ap = argparse.ArgumentParser()

    ap.add_argument("--action", required=True, choices=list(ALIAS_TO_INTERNAL.keys()))

    ap.add_argument("--backbone_ckpt", default="", help="If empty -> auto-pick from --backbone_dir by --backbone_dim_target")
    ap.add_argument("--backbone_dir", default="E:/ReAct-IR/checkpoints/backbone")
    ap.add_argument("--backbone_dim_target", type=int, default=48)
    ap.add_argument("--backbone_pick", default="epoch", choices=["epoch", "mtime"])
    ap.add_argument("--backbone_max_scan", type=int, default=0, help="0=no limit. Use to cap first-time scan if needed.")

    ap.add_argument("--lora_ckpt", default="")
    ap.add_argument("--data_root", default="E:/ReAct-IR/data")
    ap.add_argument("--results_root", default="E:/ReAct-IR/results/lora_eval")

    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--split_mode", default="all", choices=["all", "val"])
    ap.add_argument("--val_ratio", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)

    # legacy single scale
    ap.add_argument("--scale", type=float, default=1.0)

    # NEW sweep
    ap.add_argument(
        "--eval_scales",
        type=str,
        default="",
        help='Comma-separated scales to sweep in one run, e.g. "0,1" or "0,0.5,1". If empty -> use --scale only.',
    )
    ap.add_argument(
        "--no_sweep_save",
        type=int,
        default=0,
        help="If 1: during sweep, do not save images/triplets (even if save flags are on). Useful to just print metrics fast.",
    )

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=1)

    ap.add_argument("--dim", type=int, default=48)  # overridden by ckpt cfg if present
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--volterra_rank", type=int, default=2)

    ap.add_argument("--save_images", type=int, default=1)
    ap.add_argument("--save_triplets", type=int, default=1)

    a = ap.parse_args()
    return EvalConfig(
        action_alias=a.action,
        action_internal=ALIAS_TO_INTERNAL[a.action],

        backbone_ckpt=a.backbone_ckpt,
        backbone_dir=a.backbone_dir,
        backbone_dim_target=int(a.backbone_dim_target),
        backbone_pick=a.backbone_pick,
        backbone_max_scan=int(a.backbone_max_scan),

        lora_ckpt=a.lora_ckpt,
        data_root=a.data_root,
        results_root=a.results_root,

        split=a.split,
        split_mode=a.split_mode,
        val_ratio=float(a.val_ratio),
        limit=int(a.limit),
        seed=int(a.seed),

        scale=float(a.scale),
        eval_scales=str(a.eval_scales),
        no_sweep_save=bool(int(a.no_sweep_save)),

        batch_size=int(a.batch_size),
        num_workers=int(a.num_workers),
        use_amp=bool(int(a.use_amp)),
        channels_last=bool(int(a.channels_last)),
        tf32=bool(int(a.tf32)),

        dim=int(a.dim),
        bias=bool(int(a.bias)),
        volterra_rank=int(a.volterra_rank),

        save_images=bool(int(a.save_images)),
        save_triplets=bool(int(a.save_triplets)),
    )


# ============================================================
# Evaluation core
# ============================================================
def run_eval_for_scale(
    *,
    cfg: EvalConfig,
    device: torch.device,
    loader: DataLoader,
    tb: ToolBank,
    action_internal: str,
    scale: float,
    base_out_dir: str,
    save_images: bool,
    save_triplets: bool,
) -> Dict[str, float]:
    tb.activate(action_internal, float(scale))
    tb.eval()

    # output dirs (per-scale)
    scale_tag = _safe_scale_name(scale)
    out_dir = os.path.join(base_out_dir, f"scale_{scale_tag}")
    img_dir = os.path.join(out_dir, "images")
    tri_dir = os.path.join(out_dir, "triplets")
    os.makedirs(out_dir, exist_ok=True)
    if save_images:
        os.makedirs(img_dir, exist_ok=True)
    if save_triplets:
        os.makedirs(tri_dir, exist_ok=True)

    psnr_sum, ssim_sum, cnt = 0.0, 0.0, 0
    t0 = time.time()

    pbar = tqdm(loader, ncols=140, desc=f"Eval {cfg.action_alias}/{cfg.split} scale={scale}")
    with torch.no_grad():
        for it, (inp, gt, metas) in enumerate(pbar, start=1):
            if cfg.channels_last and device.type == "cuda":
                inp = inp.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                gt = gt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                inp = inp.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

            inp_pad, pad = pad_to_multiple_of(inp, mult=8)

            # ------------------------------------------------------------
            # Sanity checks on first iter (pairing + bb(scale=0) + pred(scale))
            # ------------------------------------------------------------
            if it == 1:
                inp_u8_0 = tensor_to_img_u8(inp[0])
                gt_u8_0 = tensor_to_img_u8(gt[0])
                ps_in_gt, ss_in_gt = compute_psnr_ssim(inp_u8_0, gt_u8_0)
                print(f"[Sanity][scale={scale}] PSNR(input,gt)={ps_in_gt:.2f} SSIM={ss_in_gt:.4f}")
                print(f"[Sanity][scale={scale}] meta0 =", metas[0])

                # "true backbone-only" under ToolBank wrapping => force scale=0 for the same action
                tb.activate(action_internal, 0.0)
                with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                    bb_pad = tb.backbone(inp_pad)
                bb = unpad(bb_pad, pad).clamp(0, 1)
                bb_u8_0 = tensor_to_img_u8(bb[0])
                ps_bb, ss_bb = compute_psnr_ssim(bb_u8_0, gt_u8_0)
                print(f"[Sanity][scale={scale}] PSNR(backbone(scale=0),gt)={ps_bb:.2f} SSIM={ss_bb:.4f}")

                # pred @ current scale
                tb.activate(action_internal, float(scale))
                with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                    pred_pad = tb(inp_pad)
                pred = unpad(pred_pad, pad).clamp(0, 1)

                pred_u8_0 = tensor_to_img_u8(pred[0])
                ps_pred, ss_pred = compute_psnr_ssim(pred_u8_0, gt_u8_0)
                print(f"[Sanity][scale={scale}] PSNR(pred(scale={scale}),gt)={ps_pred:.2f} SSIM={ss_pred:.4f}")

                d_mean = (pred - bb).abs().mean().item()
                d_max = (pred - bb).abs().max().item()
                print(f"[Sanity][scale={scale}] |pred-bb| mean={d_mean:.6e} max={d_max:.6e}")

                mn, mx = _tensor_range(inp)
                print(f"[Sanity][scale={scale}] inp shape/range: {tuple(inp.shape)} {mn} {mx}")
                mn, mx = _tensor_range(bb)
                print(f"[Sanity][scale={scale}] bb  shape/range: {tuple(bb.shape)} {mn} {mx}")
                mn, mx = _tensor_range(pred)
                print(f"[Sanity][scale={scale}] pred shape/range: {tuple(pred.shape)} {mn} {mx}")
            else:
                with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                    pred_pad = tb(inp_pad)
                pred = unpad(pred_pad, pad).clamp(0, 1)

            bsz = int(inp.shape[0])
            for i in range(bsz):
                key = metas[i].get("key", f"{it:06d}_{i}")
                key = str(key).replace("\\", "/").replace(":", "").replace("/", "_")

                inp_u8 = tensor_to_img_u8(inp[i])
                gt_u8 = tensor_to_img_u8(gt[i])
                pred_u8 = tensor_to_img_u8(pred[i])

                ps, ss = compute_psnr_ssim(pred_u8, gt_u8)
                psnr_sum += ps
                ssim_sum += ss
                cnt += 1

                if save_images:
                    save_img_u8(inp_u8, os.path.join(img_dir, f"{key}_in.png"))
                    save_img_u8(pred_u8, os.path.join(img_dir, f"{key}_pred.png"))
                    save_img_u8(gt_u8, os.path.join(img_dir, f"{key}_gt.png"))

                if save_triplets:
                    save_triplet(inp[i], pred[i], gt[i], os.path.join(tri_dir, f"{key}.png"))

            avg_psnr = psnr_sum / max(1, cnt)
            avg_ssim = ssim_sum / max(1, cnt)

            elapsed = time.time() - t0
            itps = it / max(1e-9, elapsed)
            remain = (len(loader) - it) / max(1e-9, itps)
            pbar.set_postfix_str(f"avgP={avg_psnr:.2f} avgS={avg_ssim:.4f} | {format_time(elapsed)}<{format_time(remain)}")

    elapsed = time.time() - t0
    return {
        "scale": float(scale),
        "psnr": float(psnr_sum / max(1, cnt)),
        "ssim": float(ssim_sum / max(1, cnt)),
        "cnt": float(cnt),
        "elapsed_sec": float(elapsed),
        "out_dir": out_dir,
    }


# ============================================================
# Main
# ============================================================
def main():
    cfg = parse_args()
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[SKIMAGE]", "ON" if USE_SKIMAGE else "OFF (PSNR/SSIM=0)")
    print(f"[Speed] tf32={cfg.tf32} channels_last={cfg.channels_last} amp={cfg.use_amp}")
    torch.backends.cudnn.benchmark = True

    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # LoRA ckpt auto-pick (optional)
    if not cfg.lora_ckpt:
        default_save_root = "E:/ReAct-IR/checkpoints/toolbank_lora"
        auto = find_latest_ckpt_in_dir(cfg.action_alias, default_save_root)
        if auto is None:
            raise RuntimeError(f"No --lora_ckpt and no files under {default_save_root}/{cfg.action_alias}")
        cfg.lora_ckpt = auto

    # Backbone ckpt auto-pick (CACHED)
    if not cfg.backbone_ckpt:
        cfg.backbone_ckpt = pick_backbone_ckpt(
            backbone_dir=cfg.backbone_dir,
            target_dim=cfg.backbone_dim_target,
            pick_mode=cfg.backbone_pick,
            verbose=True,
            max_scan=cfg.backbone_max_scan,
        )

    # Load backbone sd once
    sd_backbone = safe_load_state_dict_from_ckpt(cfg.backbone_ckpt)
    sd_backbone = _strip_known_prefixes(sd_backbone)
    print("[BackboneSD] has patch_embed.weight =", ("patch_embed.weight" in sd_backbone))

    # Prefer ckpt cfg for exact model hyperparams (bias/volterra_rank/dim)
    ckpt_cfg = safe_load_cfg_from_ckpt(cfg.backbone_ckpt)
    if ckpt_cfg:
        dim_ckpt = ckpt_cfg.get("dim", None)
        bias_ckpt = ckpt_cfg.get("bias", None)
        vr_ckpt = ckpt_cfg.get("volterra_rank", None)
        if dim_ckpt is not None:
            cfg.dim = int(dim_ckpt)
        if bias_ckpt is not None:
            cfg.bias = bool(bias_ckpt)
        if vr_ckpt is not None:
            cfg.volterra_rank = int(vr_ckpt)
        print(f"[CKPT.cfg] dim={cfg.dim} bias={cfg.bias} volterra_rank={cfg.volterra_rank}")
    else:
        inferred_dim = infer_backbone_dim_from_sd(sd_backbone)
        if inferred_dim is not None and inferred_dim != cfg.dim:
            print(f"[AutoDim] override cfg.dim {cfg.dim} -> {inferred_dim} (from chosen backbone ckpt)")
            cfg.dim = int(inferred_dim)

    # Determine sweep scales
    scales = parse_eval_scales(cfg.eval_scales, cfg.scale)

    print(f"\n[Action] {cfg.action_alias} -> {cfg.action_internal}")
    print("[Backbone] ckpt =", cfg.backbone_ckpt)
    print("[LoRA] ckpt     =", cfg.lora_ckpt)
    print(f"[Data] root     ={cfg.data_root}")
    print(f"[Eval] split    ={cfg.split}  split_mode={cfg.split_mode}  val_ratio={cfg.val_ratio}  limit={cfg.limit}")
    print(f"[BackboneArgs] dim={cfg.dim} bias={cfg.bias} volterra_rank={cfg.volterra_rank}")
    if len(scales) > 1:
        print(f"[Sweep] eval_scales={scales} (base=first scale={scales[0]})")
    else:
        print(f"[Scale] scale={scales[0]}")
    print()

    pairs, reports = build_pairs_for_action(cfg.data_root, cfg.action_internal, cfg.split)
    for rep in reports:
        print("=== [Scan] ===")
        for k, v in rep.items():
            print(f"{k}: {v}")
        print()

    if len(pairs) == 0:
        raise RuntimeError("No pairs found. Check folder structure / split.")

    ds = PairedEvalDataset(
        pairs=pairs,
        limit=cfg.limit,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
        split_mode=("val" if cfg.split_mode == "val" else "all"),
    )
    print(f"[EvalDataset] items={len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_eval,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # backbone
    backbone = VETNet(dim=cfg.dim, bias=cfg.bias, volterra_rank=cfg.volterra_rank).to(device)
    missing, unexpected = backbone.load_state_dict(sd_backbone, strict=False)
    print(f"[Backbone] loaded (missing={len(missing)} unexpected={len(unexpected)})")
    if len(missing) > 0:
        print("[Backbone] missing sample:", missing[:20])
    if len(unexpected) > 0:
        print("[Backbone] unexpected sample:", unexpected[:20])

    # toolbank
    tb = ToolBank(
        backbone=backbone,
        actions=INTERNAL_ACTIONS,
        rank=2,            # must match LoRA training
        alpha=1.0,         # must match LoRA training
        wrap_only_1x1=True,
    ).to(device)

    total_params = sum(p.numel() for p in tb.parameters())
    trainable_params = sum(p.numel() for p in tb.parameters() if p.requires_grad)
    print(f"[ToolBank] total_params={total_params/1e6:.2f}M trainable_params={trainable_params/1e6:.6f}M")

    # Load LoRA once
    load_lora_into_toolbank(tb, cfg.action_internal, cfg.lora_ckpt)

    # Base output dir (shared)
    base_out_dir = os.path.join(cfg.results_root, cfg.action_alias, f"{cfg.split}_{cfg.split_mode}")
    os.makedirs(base_out_dir, exist_ok=True)

    # Run sweep
    results: List[Dict[str, float]] = []
    for s in scales:
        # saving policy during sweep
        if len(scales) > 1 and cfg.no_sweep_save:
            save_images = False
            save_triplets = False
        else:
            save_images = cfg.save_images
            save_triplets = cfg.save_triplets

        print(f"\n====================")
        print(f"[RUN] action={cfg.action_alias} split={cfg.split} split_mode={cfg.split_mode} scale={s}")
        print(f"[SAVE] images={int(save_images)} triplets={int(save_triplets)}")
        print("====================")

        r = run_eval_for_scale(
            cfg=cfg,
            device=device,
            loader=loader,
            tb=tb,
            action_internal=cfg.action_internal,
            scale=float(s),
            base_out_dir=base_out_dir,
            save_images=save_images,
            save_triplets=save_triplets,
        )
        results.append(r)

        print("\n--------------------")
        print(f"[DONE] scale={s}")
        print(f"[AVG] PSNR={r['psnr']:.4f}  SSIM={r['ssim']:.6f}  (N={int(r['cnt'])})")
        print("[OUT] ", r["out_dir"])
        print("--------------------\n")

    # Summary + deltas
    if results:
        base = results[0]
        base_scale = base["scale"]
        base_psnr = base["psnr"]
        base_ssim = base["ssim"]

        print("\n====================")
        print(f"[SUMMARY] action={cfg.action_alias} split={cfg.split} split_mode={cfg.split_mode}")
        print(f"[BASE] scale={base_scale}  PSNR={base_psnr:.4f}  SSIM={base_ssim:.6f}")
        print("--------------------")
        for r in results:
            dps = r["psnr"] - base_psnr
            dss = r["ssim"] - base_ssim
            print(
                f"scale={r['scale']:<7g} "
                f"PSNR={r['psnr']:.4f} (Δ{dps:+.4f})  "
                f"SSIM={r['ssim']:.6f} (Δ{dss:+.6f})  "
                f"out={r['out_dir']}"
            )
        print("====================\n")


if __name__ == "__main__":
    main()
 """

# 그 LoRA가 backbone 위에서 “제대로 적용되어 추론 성능을 내는지”를 검증

"""
python -u e:/ReAct-IR/scripts/eval_lora_action.py `
>>   --action desnow `
>>   --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `        
>>   --lora_ckpt "E:/ReAct-IR/checkpoints/toolbank_lora/desnow/epoch_004_L0.0232_P30.64_S0.9411.pth" `
>>   --data_root "E:/ReAct-IR/data" `
>>   --results_root "E:/ReAct-IR/results/lora_eval" `
>>   --split test --batch_size 1 --num_workers 0 --use_amp 1 `
>>   --save_images 0 --save_triplets 0 `
>>   --scale 1

PSNR(backbone(scale=0),gt)=26.78 SSIM=0.9362

PSNR(pred(scale=1.0),gt)=28.18 SSIM=0.9594

|pred-bb| mean/max가 0이 아님 → LoRA가 forward를 실제로 바꿈


====
# desnow
python -u e:/ReAct-IR/scripts/eval_lora_action.py `
  --action desnow `
  --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
  --lora_ckpt "E:/ReAct-IR/checkpoints/toolbank_lora/desnow/epoch_004_L0.0232_P30.64_S0.9411.pth" `
  --data_root "E:/ReAct-IR/data" `
  --results_root "E:/ReAct-IR/results/lora_eval" `
  --split test --batch_size 1 --num_workers 0 --use_amp 1 `
  --save_images 0 --save_triplets 0 `
  --eval_scales "0,1"

scale=0       PSNR=27.0787 (Δ+0.0000)  SSIM=0.901150 (Δ+0.000000)  out=E:/ReAct-IR/results/lora_eval\desnow\test_all\scale_0
scale=1       PSNR=31.2882 (Δ+4.2095)  SSIM=0.946313 (Δ+0.045164)  out=E:/ReAct-IR/results/lora_eval\desnow\test_all\scale_1


#derain
python -u e:/ReAct-IR/scripts/eval_lora_action.py `
  --action derain `
  --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
  --lora_ckpt "E:/ReAct-IR/checkpoints/toolbank_lora/derain/epoch_016.pth" `
  --data_root "E:/ReAct-IR/data" `
  --results_root "E:/ReAct-IR/results/lora_eval" `
  --split test --batch_size 1 --num_workers 0 --use_amp 1 `
  --save_images 0 --save_triplets 0 `
  --eval_scales "0,1"

scale=0       PSNR=25.7442 (Δ+0.0000)  SSIM=0.799521 (Δ+0.000000)  out=E:/ReAct-IR/results/lora_eval\derain\test_all\scale_0
scale=1       PSNR=27.9206 (Δ+2.1764)  SSIM=0.848352 (Δ+0.048830)  out=E:/ReAct-IR/results/lora_eval\derain\test_all\scale_1


#dehaze
python -u e:/ReAct-IR/scripts/eval_lora_action.py `
  --action dehaze `
  --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
  --lora_ckpt "E:/ReAct-IR/checkpoints/toolbank_lora/dehaze/epoch_004.pth" `
  --data_root "E:/ReAct-IR/data" `
  --results_root "E:/ReAct-IR/results/lora_eval" `
  --split test --batch_size 1 --num_workers 0 --use_amp 1 `
  --save_images 0 --save_triplets 0 `
  --eval_scales "0,1"

scale=0       PSNR=29.3100 (Δ+0.0000)  SSIM=0.950259 (Δ+0.000000)  out=E:/ReAct-IR/results/lora_eval\dehaze\test_all\scale_0
scale=1       PSNR=32.0667 (Δ+2.7567)  SSIM=0.967255 (Δ+0.016997)  out=E:/ReAct-IR/results/lora_eval\dehaze\test_all\scale_1


#deblur
python -u e:/ReAct-IR/scripts/eval_lora_action.py `
  --action deblur `
  --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
  --lora_ckpt "E:/ReAct-IR/checkpoints/toolbank_lora/deblur/epoch_015_L0.0354_P33.29_S0.7991.pth" `
  --data_root "E:/ReAct-IR/data" `
  --results_root "E:/ReAct-IR/results/lora_eval" `
  --split test --batch_size 1 --num_workers 0 --use_amp 1 `
  --save_images 0 --save_triplets 0 `
  --eval_scales "0,1"
"""





# dehaze에서 scale 0,1 저장
# E:\ReAct-IR\scripts\eval_lora_action.py
import os
import sys
import glob
import re
import time
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm

# ------------------------------------------------------------
# Make project import-safe
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ------------------------------------------------------------
# Optional skimage metrics
# ------------------------------------------------------------
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False


# ============================================================
# Utils
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec - 60 * m)
    h = int(m // 60)
    m = int(m - 60 * h)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def is_img(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def list_images(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    files = []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            p = os.path.join(dp, f)
            if is_img(p):
                files.append(p)
    files.sort()
    return files


def make_key_from_path(p: str, root: str) -> str:
    rp = os.path.relpath(p, root).replace("\\", "/")
    rp = os.path.splitext(rp)[0]
    return rp


def imread_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_chw_float01(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1)  # CHW


def tensor_to_img_u8(t_chw: torch.Tensor) -> np.ndarray:
    t = t_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255.0).round().astype(np.uint8)


def save_img_u8(img_u8: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8).save(path)


def save_triplet(inp_chw: torch.Tensor, pred_chw: torch.Tensor, gt_chw: torch.Tensor, path: str):
    inp = tensor_to_img_u8(inp_chw)
    pr = tensor_to_img_u8(pred_chw)
    gt = tensor_to_img_u8(gt_chw)
    h, w, _ = inp.shape
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = inp
    canvas[:, w:2 * w] = pr
    canvas[:, 2 * w:3 * w] = gt
    save_img_u8(canvas, path)


def compute_psnr_ssim(pred_u8: np.ndarray, gt_u8: np.ndarray) -> Tuple[float, float]:
    if not USE_SKIMAGE:
        return 0.0, 0.0
    ps = float(peak_signal_noise_ratio(gt_u8, pred_u8, data_range=255))
    ss = float(structural_similarity(gt_u8, pred_u8, channel_axis=2, data_range=255))
    return ps, ss


def pad_to_multiple_of(x: torch.Tensor, mult: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    pad = (0, pad_w, 0, pad_h)  # (L,R,T,B)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, pad, mode="reflect")
    return x, pad


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    l, r, t, b = pad
    if r == 0 and b == 0 and l == 0 and t == 0:
        return x
    h = x.shape[-2]
    w = x.shape[-1]
    return x[..., t : h - b, l : w - r]


def _tensor_range(x: torch.Tensor) -> Tuple[float, float]:
    x = x.detach()
    return float(x.min().item()), float(x.max().item())


def _safe_scale_name(s: float) -> str:
    # 0.5 -> "0p5", 1.0 -> "1", -0.25 -> "m0p25"
    sign = "m" if s < 0 else ""
    s = abs(float(s))
    if abs(s - int(s)) < 1e-9:
        return f"{sign}{int(s)}"
    st = f"{s:.6f}".rstrip("0").rstrip(".")
    return sign + st.replace(".", "p")


def parse_eval_scales(eval_scales: str, fallback_single: float) -> List[float]:
    if eval_scales is None or str(eval_scales).strip() == "":
        return [float(fallback_single)]
    parts = [p.strip() for p in str(eval_scales).split(",") if p.strip() != ""]
    if not parts:
        return [float(fallback_single)]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            raise ValueError(f"Invalid --eval_scales item: '{p}' (expected comma-separated floats like '0,1,0.5')")
    return out


# ============================================================
# Action mapping
# ============================================================
ALIAS_TO_INTERNAL = {
    "dedrop": "A_DEDROP",
    "desnow": "A_DESNOW",
    "derain": "A_DERAIN",
    "deblur": "A_DEBLUR",
    "dehaze": "A_DEHAZE",
}
INTERNAL_ACTIONS = ["A_DEDROP", "A_DESNOW", "A_DERAIN", "A_DEBLUR", "A_DEHAZE"]


# ============================================================
# Pair scanners (standalone)
# ============================================================
def scan_pairs_by_stem(input_root: str, gt_root: str, name: str) -> Tuple[List[Tuple[str, str, Dict]], Dict]:
    in_files = list_images(input_root)
    gt_files = list_images(gt_root)

    in_map: Dict[str, str] = {}
    gt_map: Dict[str, str] = {}

    for p in in_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in in_map:
            in_map[stem] = p

    for p in gt_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in gt_map:
            gt_map[stem] = p

    keys = sorted(set(in_map.keys()) & set(gt_map.keys()))
    pairs: List[Tuple[str, str, Dict]] = []
    for k in keys:
        pairs.append((in_map[k], gt_map[k], {"key": f"{name}_{k}", "stem": k}))

    report = {
        "name": name,
        "input_root": input_root,
        "gt_root": gt_root,
        "input_total": len(in_files),
        "gt_total": len(gt_files),
        "paired": len(pairs),
        "missing_gt": len(set(in_map.keys()) - set(gt_map.keys())),
        "missing_input": len(set(gt_map.keys()) - set(in_map.keys())),
        "dup_input_stems": max(0, len(in_files) - len(in_map)),
        "dup_gt_stems": max(0, len(gt_files) - len(gt_map)),
    }
    return pairs, report


def scan_pairs_by_relpath(input_root: str, gt_root: str, name: str) -> Tuple[List[Tuple[str, str, Dict]], Dict]:
    in_files = list_images(input_root)
    gt_files = list_images(gt_root)

    in_map: Dict[str, str] = {}
    gt_map: Dict[str, str] = {}

    for p in in_files:
        k = make_key_from_path(p, input_root)
        if k not in in_map:
            in_map[k] = p

    for p in gt_files:
        k = make_key_from_path(p, gt_root)
        if k not in gt_map:
            gt_map[k] = p

    keys = sorted(set(in_map.keys()) & set(gt_map.keys()))
    pairs: List[Tuple[str, str, Dict]] = []
    for k in keys:
        pairs.append((in_map[k], gt_map[k], {"key": f"{name}_{k}", "rel": k}))

    report = {
        "name": name,
        "input_root": input_root,
        "gt_root": gt_root,
        "input_total": len(in_files),
        "gt_total": len(gt_files),
        "paired": len(pairs),
        "missing_gt": len(set(in_map.keys()) - set(gt_map.keys())),
        "missing_input": len(set(gt_map.keys()) - set(in_map.keys())),
        "dup_input_keys": max(0, len(in_files) - len(in_map)),
        "dup_gt_keys": max(0, len(gt_files) - len(gt_map)),
    }
    return pairs, report


def build_pairs_for_action(data_root: str, action_internal: str, split: str) -> Tuple[List[Tuple[str, str, Dict]], List[Dict]]:
    data_root = data_root.replace("\\", "/")
    split = split.lower().strip()

    reports: List[Dict] = []
    pairs: List[Tuple[str, str, Dict]] = []

    if action_internal == "A_DESNOW":
        if split not in ["train", "test"]:
            raise ValueError("A_DESNOW split must be train or test")
        ip = os.path.join(data_root, "CSD", split.capitalize(), "Snow")
        gp = os.path.join(data_root, "CSD", split.capitalize(), "Gt")
        pr, rep = scan_pairs_by_stem(ip, gp, name=f"csd_{split}")
        pairs.extend(pr)
        reports.append(rep)

    elif action_internal == "A_DERAIN":
        if split not in ["train", "test"]:
            raise ValueError("A_DERAIN split must be train or test")
        ip = os.path.join(data_root, "rain100H", split, "rain")
        gp = os.path.join(data_root, "rain100H", split, "norain")
        pr, rep = scan_pairs_by_stem(ip, gp, name=f"rain100H_{split}")
        pairs.extend(pr)
        reports.append(rep)

    elif action_internal == "A_DEHAZE":
        if split not in ["train", "test"]:
            raise ValueError("A_DEHAZE split must be train or test")
        ip = os.path.join(data_root, "RESIDE-6K", split, "hazy")
        gp = os.path.join(data_root, "RESIDE-6K", split, "GT")
        pr, rep = scan_pairs_by_stem(ip, gp, name=f"reside6k_{split}")
        pairs.extend(pr)
        reports.append(rep)

    elif action_internal == "A_DEDROP":
        day_in = os.path.join(data_root, "DayRainDrop", "Drop")
        day_gt = os.path.join(data_root, "DayRainDrop", "Clear")
        pr1, rep1 = scan_pairs_by_relpath(day_in, day_gt, name="dedrop_day")
        if len(pr1) == 0:
            pr1, rep1 = scan_pairs_by_stem(day_in, day_gt, name="dedrop_day_stem")
        pairs.extend(pr1)
        reports.append(rep1)

        night_in = os.path.join(data_root, "NightRainDrop", "Drop")
        night_gt = os.path.join(data_root, "NightRainDrop", "Clear")
        pr2, rep2 = scan_pairs_by_relpath(night_in, night_gt, name="dedrop_night")
        if len(pr2) == 0:
            pr2, rep2 = scan_pairs_by_stem(night_in, night_gt, name="dedrop_night_stem")
        pairs.extend(pr2)
        reports.append(rep2)

    elif action_internal == "A_DEBLUR":
        blur_in = os.path.join(data_root, "DayRainDrop", "Blur")
        clear_gt = os.path.join(data_root, "DayRainDrop", "Clear")
        pr, rep = scan_pairs_by_relpath(blur_in, clear_gt, name="deblur_day")
        if len(pr) == 0:
            pr, rep = scan_pairs_by_stem(blur_in, clear_gt, name="deblur_day_stem")
        pairs.extend(pr)
        reports.append(rep)

    else:
        raise ValueError(f"Unknown action_internal: {action_internal}")

    return pairs, reports


# ============================================================
# Dataset
# ============================================================
class PairedEvalDataset(Dataset):
    def __init__(self, pairs, limit=0, seed=123, val_ratio=0.0, split_mode="all"):
        self.all_pairs = list(pairs)

        if limit and limit > 0:
            rng = np.random.RandomState(seed)
            idxs = rng.permutation(len(self.all_pairs))[: int(limit)]
            self.all_pairs = [self.all_pairs[i] for i in idxs]

        self.pairs = self.all_pairs
        val_ratio = float(val_ratio)
        if val_ratio > 0.0:
            rng = np.random.RandomState(seed)
            n = len(self.all_pairs)
            idxs = rng.permutation(n)
            n_val = max(1, int(round(n * val_ratio)))
            val_set = set(idxs[:n_val].tolist())
            if split_mode == "val":
                self.pairs = [p for i, p in enumerate(self.all_pairs) if i in val_set]
            else:
                self.pairs = [p for i, p in enumerate(self.all_pairs) if i not in val_set]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, gt_path, meta = self.pairs[idx]
        inp = to_chw_float01(imread_rgb(inp_path))
        gt = to_chw_float01(imread_rgb(gt_path))
        if "key" not in meta:
            meta = dict(meta)
            meta["key"] = os.path.splitext(os.path.basename(inp_path))[0]
        # NOTE: keep paths for jsonl dumping
        meta = dict(meta)
        meta["inp_path"] = inp_path
        meta["gt_path"] = gt_path
        return {"input": inp, "gt": gt, "meta": meta}


def collate_eval(batch):
    xs = torch.stack([b["input"] for b in batch], dim=0)
    ys = torch.stack([b["gt"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return xs, ys, metas


# ============================================================
# CKPT helpers + Auto-pick backbone (CACHED)
# ============================================================
def safe_load_state_dict_from_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported ckpt format: {ckpt_path}")


def safe_load_cfg_from_ckpt(ckpt_path: str) -> Dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "cfg" in obj and isinstance(obj["cfg"], dict):
        return obj["cfg"]
    return {}


def infer_backbone_dim_from_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    w = sd.get("patch_embed.weight", None)
    if w is None:
        return None
    return int(w.shape[0])


def parse_epoch_from_name(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"epoch[_\-]?(\d+)", base, flags=re.IGNORECASE)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _cache_path(backbone_dir: str) -> str:
    return os.path.join(backbone_dir, "_ckpt_dim_cache.json")


def _load_cache(backbone_dir: str) -> Dict[str, Dict]:
    cp = _cache_path(backbone_dir)
    if os.path.isfile(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(backbone_dir: str, cache: Dict[str, Dict]):
    cp = _cache_path(backbone_dir)
    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def pick_backbone_ckpt(
    backbone_dir: str,
    target_dim: int,
    pick_mode: str = "epoch",  # "epoch" | "mtime"
    verbose: bool = True,
    max_scan: int = 0,  # 0 = no limit
) -> str:
    """
    Cached scanner:
      - stores {path: {dim, epoch, mtime}} into backbone_dir/_ckpt_dim_cache.json
      - next run uses cache and only rescans new/modified files
    """
    backbone_dir = backbone_dir.replace("\\", "/")
    if not os.path.isdir(backbone_dir):
        raise RuntimeError(f"--backbone_dir not found: {backbone_dir}")

    cands = sorted(glob.glob(os.path.join(backbone_dir, "*.pth")))
    if not cands:
        raise RuntimeError(f"No .pth files in backbone_dir: {backbone_dir}")

    cache = _load_cache(backbone_dir)

    to_scan = []
    for p in cands:
        mt = os.path.getmtime(p)
        rec = cache.get(p)
        if rec is None or float(rec.get("mtime", -1)) != float(mt):
            to_scan.append(p)

    if max_scan and max_scan > 0:
        to_scan = to_scan[:max_scan]

    if verbose:
        print(f"[AutoPick] cache_entries={len(cache)} total_ckpts={len(cands)} need_scan={len(to_scan)} target_dim={target_dim}")

    if to_scan:
        for p in tqdm(to_scan, ncols=120, desc="AutoPick scan ckpts"):
            try:
                sd = safe_load_state_dict_from_ckpt(p)
                dim = infer_backbone_dim_from_sd(sd)
                ep = parse_epoch_from_name(p)
                mt = os.path.getmtime(p)
                cache[p] = {"dim": dim, "epoch": ep, "mtime": mt}
            except Exception as e:
                cache[p] = {
                    "dim": None,
                    "epoch": parse_epoch_from_name(p),
                    "mtime": os.path.getmtime(p),
                    "error": repr(e),
                }
        _save_cache(backbone_dir, cache)

    ok = []
    bad = 0
    for p in cands:
        rec = cache.get(p, {})
        dim = rec.get("dim", None)
        if dim is None:
            bad += 1
            continue
        if int(dim) != int(target_dim):
            continue
        ok.append((p, int(dim), int(rec.get("epoch", -1)), float(rec.get("mtime", 0.0))))

    if verbose:
        print(f"[AutoPick] matched={len(ok)} failed={bad}")

    if not ok:
        raise RuntimeError(f"No backbone ckpt with dim={target_dim} in {backbone_dir}")

    if pick_mode == "mtime":
        ok.sort(key=lambda x: x[3], reverse=True)
    else:
        ok.sort(key=lambda x: (x[2], x[3]), reverse=True)

    chosen = ok[0][0]
    if verbose:
        print("[AutoPick] chosen:", chosen)
    return chosen


# ============================================================
# Prefix helpers (strip)
# ============================================================
def _strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Common wrappers:
      - 'module.' (DDP)
      - 'model.'  (some checkpoints)
      - 'state_dict.' (rare)
    """
    if not isinstance(sd, dict) or not sd:
        return sd

    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if all(k.startswith(prefix) for k in d.keys()):
            return {k[len(prefix) :]: v for k, v in d.items()}
        return d

    sd2 = sd
    for p in ["module.", "model.", "state_dict."]:
        sd2 = strip_prefix(sd2, p)
    return sd2


def _maybe_add_backbone_prefix_for_lora(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    ToolBank가 기대하는 key가 'backbone.xxx.lora_A...' 형태인데,
    ckpt가 'decoder1.xxx...'처럼 backbone prefix 없이 저장된 경우가 있어서 자동 보정.
    """
    if not sd:
        return sd
    keys = list(sd.keys())
    has_backbone = any(k.startswith("backbone.") for k in keys)
    if has_backbone:
        return sd
    has_lora = any(".lora_A." in k or ".lora_B." in k for k in keys)
    if not has_lora:
        return sd
    return {("backbone." + k): v for k, v in sd.items()}


# ============================================================
# LoRA loading (patched)
# ============================================================
def load_lora_into_toolbank(tb: ToolBank, action_internal: str, lora_ckpt_path: str):
    obj = torch.load(lora_ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and ("lora_state_dict" in obj):
        sd = obj["lora_state_dict"]
    elif isinstance(obj, dict) and ("state_dict" in obj) and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    else:
        sd = obj

    sd = _strip_known_prefixes(sd)
    sd = _maybe_add_backbone_prefix_for_lora(sd)

    keys = list(sd.keys())
    n_keys = len(keys)
    n_lora = sum(1 for k in keys if (".lora_A." in k or ".lora_B." in k))
    print(f"[LoRA] sd keys={n_keys}  lora_keys={n_lora}")
    if n_keys > 0:
        print("[LoRA] key sample:", keys[:5])

    try:
        if hasattr(tb, "load_lora_state_dict_for_action"):
            info = tb.load_lora_state_dict_for_action(action_internal, sd, strict=True)
            print("[LoRA] load_lora_state_dict_for_action info:", info)
        elif hasattr(tb, "load_lora_state_dict"):
            info = tb.load_lora_state_dict(action_internal, sd, strict=True)
            print("[LoRA] load_lora_state_dict info:", info)
        else:
            raise AttributeError("ToolBank must provide load_lora_state_dict_for_action(...) or load_lora_state_dict(...)")
    except RuntimeError as e:
        msg = str(e)
        print("[LoRA] strict=True failed:", msg[:500], "..." if len(msg) > 500 else "")
        print("[LoRA] retry with strict=False (partial load) ...")
        if hasattr(tb, "load_lora_state_dict_for_action"):
            info = tb.load_lora_state_dict_for_action(action_internal, sd, strict=False)
            print("[LoRA] load_lora_state_dict_for_action info (strict=False):", info)
        else:
            info = tb.load_lora_state_dict(action_internal, sd, strict=False)
            print("[LoRA] load_lora_state_dict info (strict=False):", info)


def find_latest_ckpt_in_dir(action_alias: str, save_root: str) -> Optional[str]:
    d = os.path.join(save_root, action_alias)
    if not os.path.isdir(d):
        return None
    cands = glob.glob(os.path.join(d, "*.pth"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


# ============================================================
# Config
# ============================================================
@dataclass
class EvalConfig:
    action_alias: str
    action_internal: str

    backbone_ckpt: str
    backbone_dir: str
    backbone_dim_target: int
    backbone_pick: str
    backbone_max_scan: int

    lora_ckpt: str
    data_root: str
    results_root: str

    split: str
    split_mode: str
    val_ratio: float
    limit: int
    seed: int

    # single-scale mode (legacy)
    scale: float
    # sweep mode
    eval_scales: str
    no_sweep_save: bool

    batch_size: int
    num_workers: int
    use_amp: bool
    channels_last: bool
    tf32: bool

    dim: int
    bias: bool
    volterra_rank: int

    save_images: bool
    save_triplets: bool

    # NEW: dump per-image metrics
    save_jsonl: bool


def parse_args() -> EvalConfig:
    ap = argparse.ArgumentParser()

    ap.add_argument("--action", required=True, choices=list(ALIAS_TO_INTERNAL.keys()))

    ap.add_argument("--backbone_ckpt", default="", help="If empty -> auto-pick from --backbone_dir by --backbone_dim_target")
    ap.add_argument("--backbone_dir", default="E:/ReAct-IR/checkpoints/backbone")
    ap.add_argument("--backbone_dim_target", type=int, default=48)
    ap.add_argument("--backbone_pick", default="epoch", choices=["epoch", "mtime"])
    ap.add_argument("--backbone_max_scan", type=int, default=0, help="0=no limit. Use to cap first-time scan if needed.")

    ap.add_argument("--lora_ckpt", default="")
    ap.add_argument("--data_root", default="E:/ReAct-IR/data")
    ap.add_argument("--results_root", default="E:/ReAct-IR/results/lora_eval")

    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--split_mode", default="all", choices=["all", "val"])
    ap.add_argument("--val_ratio", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)

    # legacy single scale
    ap.add_argument("--scale", type=float, default=1.0)

    # NEW sweep
    ap.add_argument(
        "--eval_scales",
        type=str,
        default="",
        help='Comma-separated scales to sweep in one run, e.g. "0,1" or "0,0.5,1". If empty -> use --scale only.',
    )
    ap.add_argument(
        "--no_sweep_save",
        type=int,
        default=0,
        help="If 1: during sweep, do not save images/triplets (even if save flags are on). Useful to just print metrics fast.",
    )

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=1)

    ap.add_argument("--dim", type=int, default=48)  # overridden by ckpt cfg if present
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--volterra_rank", type=int, default=2)

    ap.add_argument("--save_images", type=int, default=1)
    ap.add_argument("--save_triplets", type=int, default=1)

    # NEW: per-image metrics jsonl dump
    ap.add_argument(
        "--save_jsonl",
        type=int,
        default=1,
        help="If 1: save per-image PSNR/SSIM into metrics_scale_*.jsonl under each scale folder.",
    )

    a = ap.parse_args()
    return EvalConfig(
        action_alias=a.action,
        action_internal=ALIAS_TO_INTERNAL[a.action],

        backbone_ckpt=a.backbone_ckpt,
        backbone_dir=a.backbone_dir,
        backbone_dim_target=int(a.backbone_dim_target),
        backbone_pick=a.backbone_pick,
        backbone_max_scan=int(a.backbone_max_scan),

        lora_ckpt=a.lora_ckpt,
        data_root=a.data_root,
        results_root=a.results_root,

        split=a.split,
        split_mode=a.split_mode,
        val_ratio=float(a.val_ratio),
        limit=int(a.limit),
        seed=int(a.seed),

        scale=float(a.scale),
        eval_scales=str(a.eval_scales),
        no_sweep_save=bool(int(a.no_sweep_save)),

        batch_size=int(a.batch_size),
        num_workers=int(a.num_workers),
        use_amp=bool(int(a.use_amp)),
        channels_last=bool(int(a.channels_last)),
        tf32=bool(int(a.tf32)),

        dim=int(a.dim),
        bias=bool(int(a.bias)),
        volterra_rank=int(a.volterra_rank),

        save_images=bool(int(a.save_images)),
        save_triplets=bool(int(a.save_triplets)),

        save_jsonl=bool(int(a.save_jsonl)),
    )


# ============================================================
# Evaluation core
# ============================================================
def run_eval_for_scale(
    *,
    cfg: EvalConfig,
    device: torch.device,
    loader: DataLoader,
    tb: ToolBank,
    action_internal: str,
    scale: float,
    base_out_dir: str,
    save_images: bool,
    save_triplets: bool,
) -> Dict[str, float]:
    """
    Runs full dataset eval for a single scale.
    Returns dict with avg_psnr/avg_ssim/cnt/elapsed_sec.
    Additionally dumps per-image metrics to jsonl if cfg.save_jsonl is True.
    """
    tb.activate(action_internal, float(scale))
    tb.eval()

    # output dirs (per-scale)
    scale_tag = _safe_scale_name(scale)
    out_dir = os.path.join(base_out_dir, f"scale_{scale_tag}")
    img_dir = os.path.join(out_dir, "images")
    tri_dir = os.path.join(out_dir, "triplets")
    os.makedirs(out_dir, exist_ok=True)
    if save_images:
        os.makedirs(img_dir, exist_ok=True)
    if save_triplets:
        os.makedirs(tri_dir, exist_ok=True)

    # NEW: jsonl path (per-scale)
    jsonl_path = os.path.join(out_dir, f"metrics_scale_{scale_tag}.jsonl")
    jsonl_f = None
    if cfg.save_jsonl:
        # line-buffered to avoid losing progress on interrupt
        jsonl_f = open(jsonl_path, "w", encoding="utf-8", buffering=1)

    psnr_sum, ssim_sum, cnt = 0.0, 0.0, 0
    t0 = time.time()

    pbar = tqdm(loader, ncols=140, desc=f"Eval {cfg.action_alias}/{cfg.split} scale={scale}")
    with torch.no_grad():
        for it, (inp, gt, metas) in enumerate(pbar, start=1):
            if cfg.channels_last and device.type == "cuda":
                inp = inp.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                gt = gt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                inp = inp.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

            inp_pad, pad = pad_to_multiple_of(inp, mult=8)

            # ------------------------------------------------------------
            # Sanity checks on first iter (pairing + bb(scale=0) + pred(scale))
            # ------------------------------------------------------------
            if it == 1:
                inp_u8_0 = tensor_to_img_u8(inp[0])
                gt_u8_0 = tensor_to_img_u8(gt[0])
                ps_in_gt, ss_in_gt = compute_psnr_ssim(inp_u8_0, gt_u8_0)
                print(f"[Sanity][scale={scale}] PSNR(input,gt)={ps_in_gt:.2f} SSIM={ss_in_gt:.4f}")
                print(f"[Sanity][scale={scale}] meta0 =", metas[0])

                # "true backbone-only" under ToolBank wrapping => force scale=0 for the same action
                tb.activate(action_internal, 0.0)
                with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                    bb_pad = tb.backbone(inp_pad)
                bb = unpad(bb_pad, pad).clamp(0, 1)
                bb_u8_0 = tensor_to_img_u8(bb[0])
                ps_bb, ss_bb = compute_psnr_ssim(bb_u8_0, gt_u8_0)
                print(f"[Sanity][scale={scale}] PSNR(backbone(scale=0),gt)={ps_bb:.2f} SSIM={ss_bb:.4f}")

                # pred @ current scale
                tb.activate(action_internal, float(scale))
                with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                    pred_pad = tb(inp_pad)
                pred = unpad(pred_pad, pad).clamp(0, 1)

                pred_u8_0 = tensor_to_img_u8(pred[0])
                ps_pred, ss_pred = compute_psnr_ssim(pred_u8_0, gt_u8_0)
                print(f"[Sanity][scale={scale}] PSNR(pred(scale={scale}),gt)={ps_pred:.2f} SSIM={ss_pred:.4f}")

                d_mean = (pred - bb).abs().mean().item()
                d_max = (pred - bb).abs().max().item()
                print(f"[Sanity][scale={scale}] |pred-bb| mean={d_mean:.6e} max={d_max:.6e}")

                mn, mx = _tensor_range(inp)
                print(f"[Sanity][scale={scale}] inp shape/range: {tuple(inp.shape)} {mn} {mx}")
                mn, mx = _tensor_range(bb)
                print(f"[Sanity][scale={scale}] bb  shape/range: {tuple(bb.shape)} {mn} {mx}")
                mn, mx = _tensor_range(pred)
                print(f"[Sanity][scale={scale}] pred shape/range: {tuple(pred.shape)} {mn} {mx}")
            else:
                with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                    pred_pad = tb(inp_pad)
                pred = unpad(pred_pad, pad).clamp(0, 1)

            bsz = int(inp.shape[0])
            for i in range(bsz):
                meta_i = metas[i] if isinstance(metas[i], dict) else {}
                key = meta_i.get("key", f"{it:06d}_{i}")
                key = str(key)

                # for saving images filenames (sanitize)
                key_safe = key.replace("\\", "/").replace(":", "").replace("/", "_")

                inp_u8 = tensor_to_img_u8(inp[i])
                gt_u8 = tensor_to_img_u8(gt[i])
                pred_u8 = tensor_to_img_u8(pred[i])

                ps, ss = compute_psnr_ssim(pred_u8, gt_u8)
                psnr_sum += ps
                ssim_sum += ss
                cnt += 1

                if save_images:
                    save_img_u8(inp_u8, os.path.join(img_dir, f"{key_safe}_in.png"))
                    save_img_u8(pred_u8, os.path.join(img_dir, f"{key_safe}_pred.png"))
                    save_img_u8(gt_u8, os.path.join(img_dir, f"{key_safe}_gt.png"))

                if save_triplets:
                    save_triplet(inp[i], pred[i], gt[i], os.path.join(tri_dir, f"{key_safe}.png"))

                # ------------------------------------------------------------
                # NEW: dump per-image metrics (jsonl)
                # ------------------------------------------------------------
                if jsonl_f is not None:
                    inp_path = meta_i.get("inp_path", "")
                    gt_path = meta_i.get("gt_path", "")
                    rec = {
                        "id": key,           # stable id used for matching across scales
                        "inp": inp_path,
                        "gt": gt_path,
                        "psnr": float(ps),
                        "ssim": float(ss),
                        "action": cfg.action_alias,
                        "scale": float(scale),
                    }
                    jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            avg_psnr = psnr_sum / max(1, cnt)
            avg_ssim = ssim_sum / max(1, cnt)

            elapsed = time.time() - t0
            itps = it / max(1e-9, elapsed)
            remain = (len(loader) - it) / max(1e-9, itps)
            pbar.set_postfix_str(f"avgP={avg_psnr:.2f} avgS={avg_ssim:.4f} | {format_time(elapsed)}<{format_time(remain)}")

    if jsonl_f is not None:
        jsonl_f.close()
        print(f"[JSONL] saved per-image metrics: {jsonl_path}")

    elapsed = time.time() - t0
    return {
        "scale": float(scale),
        "psnr": float(psnr_sum / max(1, cnt)),
        "ssim": float(ssim_sum / max(1, cnt)),
        "cnt": float(cnt),
        "elapsed_sec": float(elapsed),
        "out_dir": out_dir,
    }


# ============================================================
# Main
# ============================================================
def main():
    cfg = parse_args()
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[SKIMAGE]", "ON" if USE_SKIMAGE else "OFF (PSNR/SSIM=0)")
    print(f"[Speed] tf32={cfg.tf32} channels_last={cfg.channels_last} amp={cfg.use_amp}")
    torch.backends.cudnn.benchmark = True

    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # LoRA ckpt auto-pick (optional)
    if not cfg.lora_ckpt:
        default_save_root = "E:/ReAct-IR/checkpoints/toolbank_lora"
        auto = find_latest_ckpt_in_dir(cfg.action_alias, default_save_root)
        if auto is None:
            raise RuntimeError(f"No --lora_ckpt and no files under {default_save_root}/{cfg.action_alias}")
        cfg.lora_ckpt = auto

    # Backbone ckpt auto-pick (CACHED)
    if not cfg.backbone_ckpt:
        cfg.backbone_ckpt = pick_backbone_ckpt(
            backbone_dir=cfg.backbone_dir,
            target_dim=cfg.backbone_dim_target,
            pick_mode=cfg.backbone_pick,
            verbose=True,
            max_scan=cfg.backbone_max_scan,
        )

    # Load backbone sd once
    sd_backbone = safe_load_state_dict_from_ckpt(cfg.backbone_ckpt)
    sd_backbone = _strip_known_prefixes(sd_backbone)
    print("[BackboneSD] has patch_embed.weight =", ("patch_embed.weight" in sd_backbone))

    # Prefer ckpt cfg for exact model hyperparams (bias/volterra_rank/dim)
    ckpt_cfg = safe_load_cfg_from_ckpt(cfg.backbone_ckpt)
    if ckpt_cfg:
        dim_ckpt = ckpt_cfg.get("dim", None)
        bias_ckpt = ckpt_cfg.get("bias", None)
        vr_ckpt = ckpt_cfg.get("volterra_rank", None)
        if dim_ckpt is not None:
            cfg.dim = int(dim_ckpt)
        if bias_ckpt is not None:
            cfg.bias = bool(bias_ckpt)
        if vr_ckpt is not None:
            cfg.volterra_rank = int(vr_ckpt)
        print(f"[CKPT.cfg] dim={cfg.dim} bias={cfg.bias} volterra_rank={cfg.volterra_rank}")
    else:
        inferred_dim = infer_backbone_dim_from_sd(sd_backbone)
        if inferred_dim is not None and inferred_dim != cfg.dim:
            print(f"[AutoDim] override cfg.dim {cfg.dim} -> {inferred_dim} (from chosen backbone ckpt)")
            cfg.dim = int(inferred_dim)

    # Determine sweep scales
    scales = parse_eval_scales(cfg.eval_scales, cfg.scale)

    print(f"\n[Action] {cfg.action_alias} -> {cfg.action_internal}")
    print("[Backbone] ckpt =", cfg.backbone_ckpt)
    print("[LoRA] ckpt     =", cfg.lora_ckpt)
    print(f"[Data] root     ={cfg.data_root}")
    print(f"[Eval] split    ={cfg.split}  split_mode={cfg.split_mode}  val_ratio={cfg.val_ratio}  limit={cfg.limit}")
    print(f"[BackboneArgs] dim={cfg.dim} bias={cfg.bias} volterra_rank={cfg.volterra_rank}")
    if len(scales) > 1:
        print(f"[Sweep] eval_scales={scales} (base=first scale={scales[0]})")
    else:
        print(f"[Scale] scale={scales[0]}")
    print(f"[JSONL] save_jsonl={int(cfg.save_jsonl)}")
    print()

    pairs, reports = build_pairs_for_action(cfg.data_root, cfg.action_internal, cfg.split)
    for rep in reports:
        print("=== [Scan] ===")
        for k, v in rep.items():
            print(f"{k}: {v}")
        print()

    if len(pairs) == 0:
        raise RuntimeError("No pairs found. Check folder structure / split.")

    ds = PairedEvalDataset(
        pairs=pairs,
        limit=cfg.limit,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
        split_mode=("val" if cfg.split_mode == "val" else "all"),
    )
    print(f"[EvalDataset] items={len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_eval,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # backbone
    backbone = VETNet(dim=cfg.dim, bias=cfg.bias, volterra_rank=cfg.volterra_rank).to(device)
    missing, unexpected = backbone.load_state_dict(sd_backbone, strict=False)
    print(f"[Backbone] loaded (missing={len(missing)} unexpected={len(unexpected)})")
    if len(missing) > 0:
        print("[Backbone] missing sample:", missing[:20])
    if len(unexpected) > 0:
        print("[Backbone] unexpected sample:", unexpected[:20])

    # toolbank
    tb = ToolBank(
        backbone=backbone,
        actions=INTERNAL_ACTIONS,
        rank=2,            # must match LoRA training
        alpha=1.0,         # must match LoRA training
        wrap_only_1x1=True,
    ).to(device)

    total_params = sum(p.numel() for p in tb.parameters())
    trainable_params = sum(p.numel() for p in tb.parameters() if p.requires_grad)
    print(f"[ToolBank] total_params={total_params/1e6:.2f}M trainable_params={trainable_params/1e6:.6f}M")

    # Load LoRA once
    load_lora_into_toolbank(tb, cfg.action_internal, cfg.lora_ckpt)

    # Base output dir (shared)
    base_out_dir = os.path.join(cfg.results_root, cfg.action_alias, f"{cfg.split}_{cfg.split_mode}")
    os.makedirs(base_out_dir, exist_ok=True)

    # Run sweep
    results: List[Dict[str, float]] = []
    for s in scales:
        # saving policy during sweep
        if len(scales) > 1 and cfg.no_sweep_save:
            save_images = False
            save_triplets = False
        else:
            save_images = cfg.save_images
            save_triplets = cfg.save_triplets

        # if jsonl saving is enabled, we still want it even when no_sweep_save=1
        print(f"\n====================")
        print(f"[RUN] action={cfg.action_alias} split={cfg.split} split_mode={cfg.split_mode} scale={s}")
        print(f"[SAVE] images={int(save_images)} triplets={int(save_triplets)} jsonl={int(cfg.save_jsonl)}")
        print("====================")

        r = run_eval_for_scale(
            cfg=cfg,
            device=device,
            loader=loader,
            tb=tb,
            action_internal=cfg.action_internal,
            scale=float(s),
            base_out_dir=base_out_dir,
            save_images=save_images,
            save_triplets=save_triplets,
        )
        results.append(r)

        print("\n--------------------")
        print(f"[DONE] scale={s}")
        print(f"[AVG] PSNR={r['psnr']:.4f}  SSIM={r['ssim']:.6f}  (N={int(r['cnt'])})")
        print("[OUT] ", r["out_dir"])
        print("--------------------\n")

    # Summary + deltas
    if results:
        base = results[0]
        base_scale = base["scale"]
        base_psnr = base["psnr"]
        base_ssim = base["ssim"]

        print("\n====================")
        print(f"[SUMMARY] action={cfg.action_alias} split={cfg.split} split_mode={cfg.split_mode}")
        print(f"[BASE] scale={base_scale}  PSNR={base_psnr:.4f}  SSIM={base_ssim:.6f}")
        print("--------------------")
        for r in results:
            dps = r["psnr"] - base_psnr
            dss = r["ssim"] - base_ssim
            print(
                f"scale={r['scale']:<7g} "
                f"PSNR={r['psnr']:.4f} (Δ{dps:+.4f})  "
                f"SSIM={r['ssim']:.6f} (Δ{dss:+.6f})  "
                f"out={r['out_dir']}"
            )
        print("====================\n")


if __name__ == "__main__":
    main()


# python E:\ReAct-IR\scripts\eval_lora_action.py --action dehaze --split test --eval_scales "0,1" --no_sweep_save 1 --save_jsonl 1
# python E:\ReAct-IR\scripts\eval_lora_action.py --action derain --split test --eval_scales "0,1" --no_sweep_save 1 --save_jsonl 1
# python E:\ReAct-IR\scripts\eval_lora_action.py --action desnow --split test --eval_scales "0,1" --no_sweep_save 1 --save_jsonl 1
# python E:\ReAct-IR\scripts\eval_lora_action.py --action deblur --split test --eval_scales "0,1" --no_sweep_save 1 --save_jsonl 1
# python E:\ReAct-IR\scripts\eval_lora_action.py --action dedrop --split test --eval_scales "0,1" --no_sweep_save 1 --save_jsonl 1