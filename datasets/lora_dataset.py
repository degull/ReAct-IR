# datasets/lora_dataset.py
# ------------------------------------------------------------
# Action LoRA training datasets (paired loading + crop/aug)
#
# Supports actions:
#   A_DEDROP : DayRainDrop(Drop/Clear) + NightRainDrop(Drop/Clear)
#   A_DEBLUR : DayRainDrop(Blur/Clear)
#   A_DESNOW : CSD(Train/Snow, Train/Gt)
#   A_DERAIN : Rain100H(train/rain, train/norain)
#   A_DEHAZE : RESIDE-6K(train/hazy, train/GT)
#
# Features:
# - Uses datasets/pairs.py scanners for pairing
# - Random crop (reflect pad) + flip/rotate augmentation
# - Returns dict: {"input": CHW float[0,1], "gt": CHW float[0,1], "meta": {...}}
# - Includes __main__ sanity prints (pair counts, sample tensor stats, preview save)
# ------------------------------------------------------------

import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# Make project import-safe (so "from datasets.pairs" works when running this file directly)
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))              # .../datasets
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))       # .../ReAct-IR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.pairs import (
    scan_raindrop_pairs,
    scan_deblur_pairs_from_raindrop,
    scan_csd_pairs,
    scan_rain100_pairs,
    scan_reside_pairs,
    PairReport,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ============================================================
# Utils
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_rgb(path: str) -> np.ndarray:
    # PIL handles tif/jpg/png; always convert to RGB
    return np.array(Image.open(path).convert("RGB"))


def to_chw_float01(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1).contiguous()


def tensor_to_img_u8(t_chw: torch.Tensor) -> np.ndarray:
    t = t_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255.0).round().astype(np.uint8)


def save_triplet(inp_chw: torch.Tensor, pred_chw: torch.Tensor, gt_chw: torch.Tensor, path: str):
    inp = tensor_to_img_u8(inp_chw)
    pr = tensor_to_img_u8(pred_chw)
    gt = tensor_to_img_u8(gt_chw)

    h, w, _ = inp.shape
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = inp
    canvas[:, w:2 * w] = pr
    canvas[:, 2 * w:3 * w] = gt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)


def _random_crop_pair(inp: np.ndarray, gt: np.ndarray, patch: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = inp.shape[:2]
    if h < patch or w < patch:
        pad_h = max(0, patch - h)
        pad_w = max(0, patch - w)
        inp = np.pad(inp, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        gt = np.pad(gt, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w = inp.shape[:2]

    y0 = random.randint(0, h - patch)
    x0 = random.randint(0, w - patch)
    return inp[y0:y0 + patch, x0:x0 + patch], gt[y0:y0 + patch, x0:x0 + patch]


def _augment_pair(inp: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # hflip
    if random.random() < 0.5:
        inp = inp[:, ::-1].copy()
        gt = gt[:, ::-1].copy()
    # vflip
    if random.random() < 0.5:
        inp = inp[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()
    # rot90
    k = random.randint(0, 3)
    if k:
        inp = np.rot90(inp, k, axes=(0, 1)).copy()
        gt = np.rot90(gt, k, axes=(0, 1)).copy()
    return inp, gt


# ============================================================
# Pair config / builder
# ============================================================
@dataclass
class ActionPairConfig:
    action: str
    split: str               # "train" or "test" (we only train with train)
    data_root: str           # E:/ReAct-IR/data
    include_day_raindrop: bool = True
    include_night_raindrop: bool = True
    max_examples_report: int = 5


def build_action_pairs(cfg: ActionPairConfig) -> Tuple[List[Tuple[str, str, Dict]], Dict[str, PairReport]]:
    """
    Returns:
      pairs: list of (inp_path, gt_path, meta)
      reports: per-source PairReport
    """
    action = cfg.action
    split = cfg.split.lower()
    if split not in ("train", "test"):
        raise ValueError(f"split must be train|test, got {cfg.split}")

    root = cfg.data_root.replace("\\", "/")
    reports: Dict[str, PairReport] = {}
    pairs_all: List[Tuple[str, str, Dict]] = []

    def _add_pairs(pairs_key: str, pairs: List[Tuple[str, str, str]], rep: PairReport, dataset_tag: str):
        reports[pairs_key] = rep
        # pairs from scanners are (key, inp, gt) — convert to (inp, gt, meta)
        for key, ip, gp in pairs:
            meta = {
                "action": action,
                "split": split,
                "dataset": dataset_tag,
                "key": key,
                "input_path": ip,
                "gt_path": gp,
            }
            pairs_all.append((ip, gp, meta))

    # ---------------- A_DEDROP ----------------
    if action == "A_DEDROP":
        if cfg.include_day_raindrop:
            drop = f"{root}/DayRainDrop/Drop"
            clear = f"{root}/DayRainDrop/Clear"
            pairs, rep = scan_raindrop_pairs("deraindrop_day", drop, clear, max_examples=cfg.max_examples_report)
            _add_pairs("deraindrop_day", pairs, rep, "DayRainDrop")
        if cfg.include_night_raindrop:
            drop = f"{root}/NightRainDrop/Drop"
            clear = f"{root}/NightRainDrop/Clear"
            pairs, rep = scan_raindrop_pairs("deraindrop_night", drop, clear, max_examples=cfg.max_examples_report)
            _add_pairs("deraindrop_night", pairs, rep, "NightRainDrop")

    # ---------------- A_DEBLUR ----------------
    elif action == "A_DEBLUR":
        blur = f"{root}/DayRainDrop/Blur"
        clear = f"{root}/DayRainDrop/Clear"
        pairs, rep = scan_deblur_pairs_from_raindrop("deblur_dayraindrop", blur, clear, max_examples=cfg.max_examples_report)
        _add_pairs("deblur_dayraindrop", pairs, rep, "DayRainDrop")

    # ---------------- A_DESNOW ----------------
    elif action == "A_DESNOW":
        csd = f"{root}/CSD"
        if split == "train":
            snow = f"{csd}/Train/Snow"
            gt = f"{csd}/Train/Gt"
            pairs, rep = scan_csd_pairs("desnow_csd_train", snow, gt, max_examples=cfg.max_examples_report)
            _add_pairs("desnow_csd_train", pairs, rep, "CSD")
        else:
            snow = f"{csd}/Test/Snow"
            gt = f"{csd}/Test/Gt"
            pairs, rep = scan_csd_pairs("desnow_csd_test", snow, gt, max_examples=cfg.max_examples_report)
            _add_pairs("desnow_csd_test", pairs, rep, "CSD")

    # ---------------- A_DERAIN ----------------
    elif action == "A_DERAIN":
        r = f"{root}/rain100H"
        if split == "train":
            rain = f"{r}/train/rain"
            gt = f"{r}/train/norain"
            pairs, rep = scan_rain100_pairs("derain_rain100H_train", rain, gt, max_examples=cfg.max_examples_report)
            _add_pairs("derain_rain100H_train", pairs, rep, "rain100H")
        else:
            rain = f"{r}/test/rain"
            gt = f"{r}/test/norain"
            pairs, rep = scan_rain100_pairs("derain_rain100H_test", rain, gt, max_examples=cfg.max_examples_report)
            _add_pairs("derain_rain100H_test", pairs, rep, "rain100H")

    # ---------------- A_DEHAZE ----------------
    elif action == "A_DEHAZE":
        r = f"{root}/RESIDE-6K"
        if split == "train":
            hazy = f"{r}/train/hazy"
            gt = f"{r}/train/GT"
            pairs, rep = scan_reside_pairs("dehaze_reside6k_train", hazy, gt, max_examples=cfg.max_examples_report)
            _add_pairs("dehaze_reside6k_train", pairs, rep, "RESIDE-6K")
        else:
            hazy = f"{r}/test/hazy"
            gt = f"{r}/test/GT"
            pairs, rep = scan_reside_pairs("dehaze_reside6k_test", hazy, gt, max_examples=cfg.max_examples_report)
            _add_pairs("dehaze_reside6k_test", pairs, rep, "RESIDE-6K")

    else:
        raise ValueError(f"Unsupported action: {action}")

    if len(pairs_all) == 0:
        raise RuntimeError(f"No pairs built for action={action} split={split} root={root}")

    return pairs_all, reports


# ============================================================
# Dataset
# ============================================================
class LoRAPairedDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str, Dict]],
        patch: int = 256,
        augment: bool = True,
        strict_size_check: bool = False,
    ):
        """
        pairs: [(input_path, gt_path, meta), ...]
        patch: random crop size
        augment: flip/rotate
        strict_size_check: if True, asserts input/gt sizes match before crop
        """
        self.pairs = pairs
        self.patch = int(patch)
        self.augment = bool(augment)
        self.strict_size_check = bool(strict_size_check)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        ip, gp, meta = self.pairs[idx]
        inp = imread_rgb(ip)
        gt = imread_rgb(gp)

        if self.strict_size_check and (inp.shape[:2] != gt.shape[:2]):
            raise RuntimeError(f"Size mismatch: inp={inp.shape} gt={gt.shape}\n  inp={ip}\n  gt={gp}")

        inp, gt = _random_crop_pair(inp, gt, self.patch)
        if self.augment:
            inp, gt = _augment_pair(inp, gt)

        inp_t = to_chw_float01(inp)
        gt_t = to_chw_float01(gt)
        return {"input": inp_t, "gt": gt_t, "meta": meta}


def collate_lora(batch: List[Dict]):
    xs = torch.stack([b["input"] for b in batch], dim=0)
    ys = torch.stack([b["gt"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return xs, ys, metas


# ============================================================
# Debug / sanity check
# ============================================================
def _print_reports(reports: Dict[str, PairReport]):
    for k in sorted(reports.keys()):
        rep = reports[k]
        print(f"\n=== [{rep.name}] ===")
        print(f"input_root: {rep.input_root}")
        print(f"gt_root   : {rep.gt_root}")
        print(f"input_total={rep.input_total}  gt_total={rep.gt_total}")
        print(f"paired={rep.paired}  missing_gt={rep.missing_gt}  missing_input={rep.missing_input}")
        print(f"dup_input_keys={rep.dup_input_keys}  dup_gt_keys={rep.dup_gt_keys}")


def _debug_run_one_action(
    action: str,
    split: str = "train",
    data_root: str = "E:/ReAct-IR/data",
    patch: int = 256,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 123,
    save_preview: bool = True,
):
    seed_all(seed)

    cfg = ActionPairConfig(action=action, split=split, data_root=data_root)
    pairs, reports = build_action_pairs(cfg)

    print("\n--------------------------------------------------")
    print(f"[lora_dataset.py][DEBUG] action={action} split={split}")
    print(f"[lora_dataset.py][DEBUG] data_root={data_root}")
    print(f"[lora_dataset.py][DEBUG] total_pairs={len(pairs)} patch={patch} augment=True")
    _print_reports(reports)

    ds = LoRAPairedDataset(pairs=pairs, patch=patch, augment=True, strict_size_check=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_lora,
        persistent_workers=(num_workers > 0),
    )

    xs, ys, metas = next(iter(loader))
    print("\n[lora_dataset.py][DEBUG] batch shapes:")
    print("  input:", tuple(xs.shape), xs.dtype, "min/max:", float(xs.min()), float(xs.max()))
    print("  gt   :", tuple(ys.shape), ys.dtype, "min/max:", float(ys.min()), float(ys.max()))
    print("  meta[0]:", metas[0])

    # Save a quick preview triplet (inp | inp | gt) just to verify decoding/cropping visually
    if save_preview:
        out_dir = os.path.join(PROJECT_ROOT, "results", "debug_lora_dataset")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"preview_{action}_{split}.png")
        save_triplet(xs[0], xs[0], ys[0], out_path)
        print("[lora_dataset.py][DEBUG] saved preview:", out_path.replace("\\", "/"))


if __name__ == "__main__":
    # Run a few sanity checks. You can change these locally.
    _debug_run_one_action("A_DESNOW", split="train", data_root="E:/ReAct-IR/data", patch=256, batch_size=2)
    _debug_run_one_action("A_DEDROP", split="train", data_root="E:/ReAct-IR/data", patch=256, batch_size=2)
    _debug_run_one_action("A_DERAIN", split="train", data_root="E:/ReAct-IR/data", patch=256, batch_size=2)
    print("\n[lora_dataset.py][DEBUG] OK")
