# scripts/eval_toolbank_sanity.py
import os
import sys
import random
import gc
from typing import List, Dict, Any

# --------------------------------------------------
# (IMPORTANT) Set allocator config BEFORE importing torch
#   - helps fragmentation when loading large checkpoints on CUDA
# --------------------------------------------------
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Imports (project)
# --------------------------------------------------
from datasets.mixed_dataset import MixedDataset
from datasets.csd import CSDDataset
from datasets.rain100 import Rain100Dataset
from datasets.raindrop_day import DayRainDropDataset
from datasets.raindrop_night import NightRainDropDataset
from datasets.reside6k import RESIDE6KDataset

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank, AdapterSpec
from models.planner.action_space import (
    A_DEDROP, A_DEBLUR, A_DESNOW, A_DERAIN, A_DEHAZE, A_HYBRID, A_STOP
)

# --------------------------------------------------
# Utils
# --------------------------------------------------
import yaml
import numpy as np
from PIL import Image


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Paired Transform (same as train)
# --------------------------------------------------
def _pil_to_tensor(img: Image.Image):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _random_crop_pair(inp, gt, patch):
    w, h = inp.size
    if w < patch or h < patch:
        scale = max(patch / w, patch / h)
        nw, nh = int(w * scale), int(h * scale)
        inp = inp.resize((nw, nh), Image.BILINEAR)
        gt = gt.resize((nw, nh), Image.BILINEAR)
        w, h = inp.size
    x = random.randint(0, w - patch)
    y = random.randint(0, h - patch)
    return inp.crop((x, y, x + patch, y + patch)), gt.crop((x, y, x + patch, y + patch))


def _augment_pair(inp, gt):
    if random.random() < 0.5:
        inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    k = random.randint(0, 3)
    if k > 0:
        inp = inp.rotate(90 * k, expand=True)
        gt = gt.rotate(90 * k, expand=True)
    return inp, gt


class PairedTransform:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

    def __call__(self, inp, gt):
        inp, gt = _random_crop_pair(inp, gt, self.patch_size)
        inp, gt = _augment_pair(inp, gt)
        return _pil_to_tensor(inp), _pil_to_tensor(gt)


# --------------------------------------------------
# Oracle Action (same as train)
# --------------------------------------------------
def choose_oracle_action(degs: List[str]) -> str:
    d = set(x.lower() for x in degs)
    if "drop" in d:
        return A_DEDROP
    if "rain" in d:
        return A_DERAIN
    if "haze" in d:
        return A_DEHAZE
    if "snow" in d:
        return A_DESNOW
    if "blur" in d:
        return A_DEBLUR
    return A_HYBRID


# --------------------------------------------------
# Custom Collate (same as train)
# --------------------------------------------------
def react_ir_collate_fn(batch):
    return {
        "input": torch.stack([b["input"] for b in batch], dim=0),
        "gt": torch.stack([b["gt"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


# --------------------------------------------------
# Dataset Factory (same as train)
# --------------------------------------------------
def build_single_dataset(root, cfg, tfm):
    t = cfg["type"]
    split = cfg.get("split", "train")
    if t == "CSDDataset":
        return CSDDataset(root=root, split=split, transform=tfm, debug=False)
    if t == "DayRainDropDataset":
        return DayRainDropDataset(root=root, split=split, transform=tfm, debug=False)
    if t == "NightRainDropDataset":
        return NightRainDropDataset(root=root, split=split, transform=tfm, debug=False)
    if t == "Rain100Dataset":
        return Rain100Dataset(root=root, split=split, transform=tfm, debug=False)
    if t == "RESIDE6KDataset":
        return RESIDE6KDataset(root=root, split=split, transform=tfm, debug=False)
    raise ValueError(t)


def build_mixed_dataset(cfg, tfm):
    root = cfg["data_root"]
    mixed = cfg["mixed"]
    datasets = [build_single_dataset(root, dcfg, tfm) for dcfg in cfg["datasets"].values()]
    return MixedDataset(
        datasets=datasets,
        balance=mixed.get("balance", "sqrt"),
        epoch_length=int(mixed.get("epoch_length", 20000)),
        seed=int(mixed.get("seed", 123)),
        debug=True,
    )


# --------------------------------------------------
# Sanity metrics (eval-only)
# --------------------------------------------------
@torch.no_grad()
def _forward_with_action(toolbank: ToolBank, x1: torch.Tensor, action: str, use_amp: bool = True) -> torch.Tensor:
    toolbank.activate_adapter(action if action != A_STOP else A_STOP)
    if use_amp and x1.is_cuda:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return toolbank.backbone(x1)
    return toolbank.backbone(x1)


@torch.no_grad()
def compute_pair_metrics(
    toolbank: ToolBank,
    x: torch.Tensor,
    sel_action: str,
    neg_action: str,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Returns:
      - delta_sel_vs_stop = mean(|f_sel(x) - f_stop(x)|)
      - delta_sel_vs_neg  = mean(|f_sel(x) - f_neg(x)|)
      - sep_score         = delta_sel_vs_neg / (delta_sel_vs_stop + eps)
    """
    eps = 1e-8
    x1 = x[:1]  # 1-sample only (memory safe)

    out_stop = _forward_with_action(toolbank, x1, A_STOP, use_amp=use_amp)
    out_sel = _forward_with_action(toolbank, x1, sel_action, use_amp=use_amp)
    out_neg = _forward_with_action(toolbank, x1, neg_action, use_amp=use_amp)

    delta_sel_vs_stop = float((out_sel - out_stop).abs().mean().item())
    delta_sel_vs_neg = float((out_sel - out_neg).abs().mean().item())
    sep_score = float(delta_sel_vs_neg / (delta_sel_vs_stop + eps))

    return {
        "delta_sel_vs_stop": delta_sel_vs_stop,
        "delta_sel_vs_neg": delta_sel_vs_neg,
        "sep_score": sep_score,
    }


# --------------------------------------------------
# Robust checkpoint loader
#   - tries CPU first (if you have enough RAM)
#   - if CPU fails due to RAM, falls back to CUDA load
# --------------------------------------------------
def load_checkpoint_safely(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # reduce fragmentation before loading
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 1) Try CPU load (preferred if you have enough RAM)
    try:
        print(f"[CKPT] Trying load on CPU: {ckpt_path}")
        # mmap=True reduces peak RAM on some torch versions
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        print("[CKPT] CPU load OK")
        return ckpt
    except RuntimeError as e:
        msg = str(e).lower()
        cpu_oom = ("defaultcpuallocator" in msg) or ("not enough memory" in msg)
        if not cpu_oom:
            raise

        print("[CKPT][WARN] CPU RAM 부족으로 CPU load 실패. CUDA로 fallback 합니다.")
        # 2) Fallback: load directly to CUDA (avoid CPU RAM)
        if device.type != "cuda":
            raise RuntimeError(
                "CPU load failed due to RAM, but CUDA is not available. "
                "Enable pagefile / free RAM / or load on a machine with more RAM."
            )

        # extra cleanup
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[CKPT] Trying load on CUDA: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        print("[CKPT] CUDA load OK")
        return ckpt


# --------------------------------------------------
# Main (EVAL ONLY)
# --------------------------------------------------
def main():
    cfg_ds = load_yaml(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    cfg_tools = load_yaml(os.path.join(PROJECT_ROOT, "configs", "tools.yaml"))

    set_seed(int(cfg_ds.get("mixed", {}).get("seed", 123)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # ---------
    # Eval settings
    # ---------
    ckpt_path = cfg_tools.get("eval", {}).get(
        "ckpt_path",
        r"E:\ReAct-IR\checkpoints\toolbank\epoch_020_loss0.0689.pth",
    )

    num_eval_batches = int(cfg_tools.get("eval", {}).get("num_batches", 5))
    use_amp = bool(cfg_tools.get("eval", {}).get("use_amp", True))

    # (0-1) requested: neg_action cycles over this set
    sanity_neg_actions = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DEHAZE]

    # thresholds
    thr_delta_stop = float(cfg_tools.get("eval", {}).get("thr_delta_stop", 1e-6))
    thr_sep = float(cfg_tools.get("eval", {}).get("thr_sep", 1.0))

    # ---------
    # Dataset
    # ---------
    tfm = PairedTransform(cfg_ds["train_patch"]["patch_size"])
    eval_set = build_mixed_dataset(cfg_ds, tfm)

    loader = DataLoader(
        eval_set,
        batch_size=cfg_ds["loader"]["batch_size"],
        shuffle=True,
        num_workers=cfg_ds["loader"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=react_ir_collate_fn,
    )

    # ---------
    # Model + ToolBank
    # ---------
    backbone = VETNet(dim=48, volterra_rank=4).to(device)
    adapter_specs = {a: AdapterSpec(**s) for a, s in cfg_tools["toolbank"]["adapters"].items()}
    toolbank = ToolBank(backbone, adapter_specs, device=device, debug=True).to(device)
    toolbank.eval()

    # ---------
    # Load checkpoint robustly
    # ---------
    print(f"[CKPT] Loading: {ckpt_path}")
    ckpt = load_checkpoint_safely(ckpt_path, device=device)

    state = ckpt["toolbank"] if isinstance(ckpt, dict) and "toolbank" in ckpt else ckpt
    missing, unexpected = toolbank.load_state_dict(state, strict=False)
    print(f"[CKPT] Loaded into ToolBank. missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  missing keys (first 20):")
        for k in missing[:20]:
            print("   -", k)
    if len(unexpected) > 0:
        print("  unexpected keys (first 20):")
        for k in unexpected[:20]:
            print("   -", k)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------
    # EVAL LOOP (NO TRAINING)
    # ---------
    agg: Dict[str, Dict[str, Any]] = {}

    def _key(sel: str, neg: str) -> str:
        return f"{sel}__VS__{neg}"

    seen_batches = 0
    for batch in loader:
        seen_batches += 1
        if seen_batches > num_eval_batches:
            break

        x = batch["input"].to(device, non_blocking=True)

        oracle_actions = [choose_oracle_action(m["degradations"]) for m in batch["meta"]]
        sel_action = max(set(oracle_actions), key=oracle_actions.count)

        print(f"\n[SANITY][pairs] batch={seen_batches}/{num_eval_batches} sel_action={sel_action}")

        for neg_action in sanity_neg_actions:
            if neg_action == sel_action:
                continue

            m = compute_pair_metrics(toolbank, x, sel_action, neg_action, use_amp=use_amp)

            k = _key(sel_action, neg_action)
            if k not in agg:
                agg[k] = {
                    "sel": sel_action,
                    "neg": neg_action,
                    "n": 0,
                    "pass_delta": 0,
                    "pass_sep": 0,
                    "sum_delta_stop": 0.0,
                    "sum_sep": 0.0,
                }

            agg[k]["n"] += 1
            agg[k]["sum_delta_stop"] += float(m["delta_sel_vs_stop"])
            agg[k]["sum_sep"] += float(m["sep_score"])

            if m["delta_sel_vs_stop"] > thr_delta_stop:
                agg[k]["pass_delta"] += 1
            if m["sep_score"] >= thr_sep:
                agg[k]["pass_sep"] += 1

            print(
                f"  - neg={neg_action:<8} "
                f"Δsel_stop={m['delta_sel_vs_stop']:.6f} "
                f"Δsel_neg={m['delta_sel_vs_neg']:.6f} "
                f"sep={m['sep_score']:.4f}"
            )

        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------
    # Summary
    # ---------
    print("\n" + "=" * 80)
    print("[SUMMARY] action-pair sanity (aggregated over evaluated batches)")
    print(f"  - ckpt: {ckpt_path}")
    print(f"  - batches: {min(seen_batches, num_eval_batches)}")
    print(f"  - thr_delta_stop: {thr_delta_stop}")
    print(f"  - thr_sep: {thr_sep}")
    print("-" * 80)

    keys_sorted = sorted(agg.keys(), key=lambda s: (agg[s]["sel"], agg[s]["neg"]))
    for k in keys_sorted:
        a = agg[k]
        n = max(1, int(a["n"]))
        mean_delta = float(a["sum_delta_stop"]) / n
        mean_sep = float(a["sum_sep"]) / n
        print(
            f"{a['sel']:<8} vs {a['neg']:<8} | "
            f"mean(Δsel_stop)={mean_delta:.6f} | "
            f"mean(sep)={mean_sep:.4f} | "
            f"PASS Δ: {a['pass_delta']}/{n} | "
            f"PASS sep: {a['pass_sep']}/{n}"
        )

    if len(agg) == 0:
        print("\n[WARN] No pairs evaluated.")
    else:
        all_delta_ok = all(a["pass_delta"] == a["n"] for a in agg.values())
        total_sep = sum(a["n"] for a in agg.values())
        total_sep_pass = sum(a["pass_sep"] for a in agg.values())
        sep_pass_ratio = float(total_sep_pass) / float(max(1, total_sep))

        print("\n[GLOBAL]")
        print(f"  - All pairs satisfy Δsel_stop > thr? : {all_delta_ok}")
        print(f"  - sep>=thr ratio (over all pair-evals): {sep_pass_ratio:.3f} ({total_sep_pass}/{total_sep})")

    print("=" * 80)
    print("[Eval] Done ✅")


if __name__ == "__main__":
    main()
