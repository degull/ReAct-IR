# scripts/train_lora_action.py
# ------------------------------------------------------------
# Train a single action LoRA in ToolBank, with frozen VETNet backbone.
#
# CLI action aliases:
#   --action {dedrop, deblur, desnow, derain, dehaze}
# Mapped internally to:
#   A_DEDROP, A_DEBLUR, A_DESNOW, A_DERAIN, A_DEHAZE
#
# Saves:
#   E:/ReAct-IR/checkpoints/toolbank_lora/{action}/epoch_XXX_....pth
# where the checkpoint contains ONLY LoRA weights for that action.
#
# Also saves preview images (inp|pred|gt) every --iter_save_interval.
#
# Includes __main__ debug checks:
# - 1 step loss(before/after) on same batch
# - saved state_dict tensor/elem counts
# - save->reload output diff check (LoRA-only fidelity)
#
# Key features in this final version:
#   (1) strict dim-check vs ckpt (detect ckpt dim from patch_embed.weight[0])
#       -> fail fast if mismatch (e.g. dim48 ckpt into dim64 model)
#   (2) compat backbone loader:
#       - accept ckpt as {"state_dict": ...} or plain state_dict
#       - strip lora keys if ckpt is ToolBank-wrapped
#       - unwrap ".base." keys if ckpt stores base weights as "*.base.weight"
#       - remap legacy names: blocks->body, volterra->volt
#       - duplicate volterra.* -> volt1/volt2 where needed
#       - print detailed load ratio
#   (3) save_reload_diff_check uses extract_pure_backbone_state_dict(base_ref)
#       and loads base2 from it (avoids "Unexpected lora_A/base" issues)
# ------------------------------------------------------------

import os
import sys
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank

from datasets.lora_dataset import (
    ActionPairConfig,
    build_action_pairs,
    LoRAPairedDataset,
    collate_lora,
    tensor_to_img_u8,  # already defined there
)

# --------------------------------------------------
# Optional skimage metrics
# --------------------------------------------------
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False


# ============================================================
# Utilities
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


def count_trainable_params(model: torch.nn.Module) -> float:
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            n += p.numel()
    return n / 1e6


def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def save_triplet(inp_chw: torch.Tensor, pred_chw: torch.Tensor, gt_chw: torch.Tensor, path: str):
    inp = tensor_to_img_u8(inp_chw)
    pr = tensor_to_img_u8(pred_chw)
    gt = tensor_to_img_u8(gt_chw)

    h, w, _ = inp.shape
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = inp
    canvas[:, w : 2 * w] = pr
    canvas[:, 2 * w : 3 * w] = gt

    safe_makedirs(os.path.dirname(path))
    Image.fromarray(canvas).save(path)


def compute_psnr_ssim_1(pred_chw: torch.Tensor, gt_chw: torch.Tensor) -> Tuple[float, float]:
    if not USE_SKIMAGE:
        return 0.0, 0.0
    p = tensor_to_img_u8(pred_chw)
    g = tensor_to_img_u8(gt_chw)
    ps = float(peak_signal_noise_ratio(g, p, data_range=255))
    ss = float(structural_similarity(g, p, channel_axis=2, data_range=255))
    return ps, ss


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def state_dict_stats(sd: Dict[str, torch.Tensor]) -> Tuple[int, float]:
    tensors = 0
    elems = 0
    for _, v in sd.items():
        if torch.is_tensor(v):
            tensors += 1
            elems += int(v.numel())
    return tensors, elems / 1e6


# ============================================================
# ToolBank LoRA SD helpers (supports multiple ToolBank APIs)
# ============================================================
def lora_state_dict_for_action(tb: ToolBank, action: str) -> Dict[str, torch.Tensor]:
    if hasattr(tb, "lora_state_dict_for_action"):
        return tb.lora_state_dict_for_action(action)
    if hasattr(tb, "lora_state_dict"):
        return tb.lora_state_dict(action)
    raise AttributeError("ToolBank must provide lora_state_dict(action) or lora_state_dict_for_action(action)")


def load_lora_state_dict_for_action(tb: ToolBank, action: str, sd: Dict[str, torch.Tensor], strict: bool = True):
    """
    Load LoRA weights for a specific action.

    Preferred:
      - tb.load_lora_state_dict_for_action(action, sd)
      - tb.load_lora_state_dict(action, sd)

    Fallback:
      - tb.load_state_dict(sd, strict=False) assuming sd keys are full module paths.
    """
    if hasattr(tb, "load_lora_state_dict_for_action"):
        return tb.load_lora_state_dict_for_action(action, sd, strict=strict)
    if hasattr(tb, "load_lora_state_dict"):
        return tb.load_lora_state_dict(action, sd, strict=strict)

    missing, unexpected = tb.load_state_dict(sd, strict=False)
    if strict and (len(unexpected) > 0):
        raise RuntimeError(
            f"Unexpected keys while loading LoRA state_dict (strict=True). "
            f"unexpected={unexpected[:10]} (total {len(unexpected)})"
        )
    return {"missing": missing, "unexpected": unexpected}


# ============================================================
# Backbone CKPT compat loader
# ============================================================
def _normalize_ckpt_to_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Accept:
      - {"state_dict": {...}}
      - {"model": {...}}
      - plain state_dict
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        # plain dict of tensors?
        if any(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj  # type: ignore
    raise RuntimeError("Unsupported checkpoint format (expected state_dict or dict with state_dict/model)")


def _detect_dim_from_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Try to infer model dim from patch_embed weights.
    VETNet.patch_embed: Conv2d(in=3,out=dim,k=3)
    weight shape: [dim, 3, 3, 3]
    """
    for k in ("patch_embed.weight", "module.patch_embed.weight"):
        if k in sd and torch.is_tensor(sd[k]):
            w = sd[k]
            if w.ndim == 4 and w.shape[1] == 3:
                return int(w.shape[0])
    return None


def _strip_lora_keys_from_sd(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    If checkpoint is a ToolBank-wrapped state_dict, it may contain:
      - ".lora_A." / ".lora_B."
      - "lora_A." / "lora_B."
    We remove those to get pure backbone keys only.
    """
    stripped = {}
    n_strip = 0
    for k, v in sd.items():
        if (".lora_A." in k) or (".lora_B." in k) or ("lora_A." in k) or ("lora_B." in k):
            n_strip += 1
            continue
        stripped[k] = v
    return stripped, {"stripped_lora": n_strip, "kept": len(stripped)}


def _unwrap_base_keys(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Some wrappers store base weights as:
      encoder1.body.0.attn.qkv.base.weight  -> encoder1.body.0.attn.qkv.weight
    We convert "*.base.(weight|bias)" -> "*(weight|bias)" if target doesn't already exist.
    """
    out = dict(sd)
    n_unwrap = 0
    for k, v in list(sd.items()):
        if ".base." not in k:
            continue
        k2 = k.replace(".base.", ".")
        if k2 not in out:
            out[k2] = v
            n_unwrap += 1
    return out, {"unwrapped_base": n_unwrap}


def _remap_key_legacy(k: str) -> Tuple[str, Dict[str, int]]:
    """
    Legacy mappings observed in your logs:
      - encoderX.blocks.N.*  -> encoderX.body.N.*
      - *.volterra.*         -> *.volt1.*  (then we may duplicate to volt2)
    """
    stats = {"blocks_to_body": 0, "volterra_to_volt": 0}

    if ".blocks." in k:
        k = k.replace(".blocks.", ".body.")
        stats["blocks_to_body"] += 1

    if ".volterra." in k:
        k = k.replace(".volterra.", ".volt1.")
        stats["volterra_to_volt"] += 1

    return k, stats


def _compat_remap_and_filter_to_model(
    sd_in: Dict[str, torch.Tensor], model: torch.nn.Module
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Produce a state_dict whose keys match the current model keys as much as possible.

    Steps:
      1) apply legacy renames (blocks->body, volterra->volt1)
      2) keep only keys that exist in model AND shapes match
      3) duplicate volt1 -> volt2 when model expects volt2 and ckpt only has volt1 (for volterra parts)
    """
    model_sd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}

    stats = {
        "blocks_to_body": 0,
        "volterra_to_volt": 0,
        "direct_kept": 0,
        "shape_mismatch_drop": 0,
        "missing_in_model_drop": 0,
        "dup_to_volt2": 0,
    }

    # 1) remap
    remapped: Dict[str, torch.Tensor] = {}
    for k, v in sd_in.items():
        k2, s = _remap_key_legacy(k)
        stats["blocks_to_body"] += s["blocks_to_body"]
        stats["volterra_to_volt"] += s["volterra_to_volt"]
        remapped[k2] = v

    # 2) keep matching keys by name & shape
    for k, v in remapped.items():
        if k not in model_sd:
            stats["missing_in_model_drop"] += 1
            continue
        if not torch.is_tensor(v) or not torch.is_tensor(model_sd[k]):
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            stats["shape_mismatch_drop"] += 1
            continue
        out[k] = v
        stats["direct_kept"] += 1

    # 3) duplicate volt1 -> volt2 if volt2 exists and missing
    # Example: encoder1.body.0.volt1.* -> encoder1.body.0.volt2.*
    for k, v in list(out.items()):
        if ".volt1." not in k:
            continue
        k2 = k.replace(".volt1.", ".volt2.")
        if k2 in model_sd and k2 not in out:
            if tuple(v.shape) == tuple(model_sd[k2].shape):
                out[k2] = v
                stats["dup_to_volt2"] += 1

    return out, stats


def load_backbone_ckpt_compat(
    model: torch.nn.Module,
    ckpt_path: str,
    expect_dim: Optional[int] = None,
    fail_fast_on_dim_mismatch: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Robust backbone loader:
      - supports ckpt dict formats
      - strips LoRA keys if present
      - unwraps base keys if present
      - remaps legacy names
      - filters by model keys & shape
      - prints load ratios

    Returns an info dict (also printed).
    """
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    raw_sd = _normalize_ckpt_to_state_dict(ckpt_obj)

    # infer ckpt dim
    ckpt_dim = _detect_dim_from_sd(raw_sd)

    if expect_dim is not None and ckpt_dim is not None and ckpt_dim != expect_dim:
        msg = f"[Backbone] DIM MISMATCH: ckpt_dim={ckpt_dim} vs model_dim={expect_dim} ({ckpt_path})"
        if fail_fast_on_dim_mismatch:
            raise RuntimeError(msg)
        else:
            print(msg)

    # strip lora keys & unwrap base keys (if any)
    sd1, strip_stats = _strip_lora_keys_from_sd(raw_sd)
    sd2, unwrap_stats = _unwrap_base_keys(sd1)

    # compat remap + shape filter
    sd3, remap_stats = _compat_remap_and_filter_to_model(sd2, model)

    # load
    missing, unexpected = model.load_state_dict(sd3, strict=False)

    # load ratios
    model_total = len(model.state_dict())
    ckpt_total = len(raw_sd)
    loaded_keys = len(sd3)

    # How many of ckpt keys did we actually use (after filtering)?
    ckpt_used_ratio = 100.0 * loaded_keys / max(1, ckpt_total)
    model_loaded_ratio = 100.0 * loaded_keys / max(1, model_total)

    info = {
        "ckpt_path": ckpt_path,
        "ckpt_total_keys": int(ckpt_total),
        "model_total_keys": int(model_total),
        "loaded_keys": int(loaded_keys),
        "model_loaded_ratio": float(model_loaded_ratio),
        "ckpt_used_ratio": float(ckpt_used_ratio),
        "missing_keys": int(len(missing)),
        "unexpected_keys": int(len(unexpected)),
        "example_missing": missing[:25],
        "example_unexpected": unexpected[:25],
        "has_state_dict": isinstance(ckpt_obj, dict) and ("state_dict" in ckpt_obj),
        "ckpt_dim": ckpt_dim,
        "CompatV2": {**strip_stats, **unwrap_stats},
        "Remap": remap_stats,
    }

    if verbose:
        ctor_dim = None
        # best-effort: read model dim from patch_embed
        try:
            ctor_dim = int(model.patch_embed.out_channels)  # type: ignore
        except Exception:
            pass

        print(f"[Backbone] ctor: dim={ctor_dim} bias={getattr(model, 'bias', 'NA')} volterra_rank=NA")
        print(f"[Backbone] ckpt = {ckpt_path}")
        if ckpt_dim is not None:
            print(f"[Backbone] ckpt_dim inferred = {ckpt_dim}")
        print(
            f"[Backbone] loaded_keys={loaded_keys}/{model_total} "
            f"({model_loaded_ratio:.2f}% of model, {ckpt_used_ratio:.2f}% of ckpt) "
            f"| missing={len(missing)} unexpected={len(unexpected)}"
        )
        print(f"  [CompatV2] stripped_lora={strip_stats['stripped_lora']} unwrapped_base={unwrap_stats['unwrapped_base']}")
        print(
            "  [Remap] "
            f"blocks->body={remap_stats['blocks_to_body']} "
            f"volterra->volt={remap_stats['volterra_to_volt']} "
            f"direct_kept={remap_stats['direct_kept']} "
            f"dup_to_volt2={remap_stats['dup_to_volt2']} "
            f"shape_drop={remap_stats['shape_mismatch_drop']} "
            f"name_drop={remap_stats['missing_in_model_drop']}"
        )

        if len(missing) > 0:
            print("  example_missing   :", info["example_missing"])
        if len(unexpected) > 0:
            print("  example_unexpected:", info["example_unexpected"])

    return info


def extract_pure_backbone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract a "pure backbone" state_dict from any module.
    Even if called on a ToolBank-wrapped module accidentally, it will:
      - strip lora keys
      - unwrap base keys
      - return only keys that look like backbone weights (heuristic: no lora_A/lora_B)
    For plain VETNet, it simply returns model.state_dict().
    """
    sd = model.state_dict()

    # If it's already pure, these ops are no-ops
    sd1, _ = _strip_lora_keys_from_sd(sd)
    sd2, _ = _unwrap_base_keys(sd1)

    # Also remove any leftover keys that still contain lora in name
    out = {}
    for k, v in sd2.items():
        if (".lora_A." in k) or (".lora_B." in k) or ("lora_A." in k) or ("lora_B." in k):
            continue
        out[k] = v
    return out


# ============================================================
# Action mapping
# ============================================================
ACTION_ALIAS = {
    "dedrop": "A_DEDROP",
    "deblur": "A_DEBLUR",
    "desnow": "A_DESNOW",
    "derain": "A_DERAIN",
    "dehaze": "A_DEHAZE",
}


def map_action(alias: str) -> Tuple[str, str]:
    a = alias.strip().lower()
    if a not in ACTION_ALIAS:
        raise ValueError(f"--action must be one of {list(ACTION_ALIAS.keys())}, got {alias}")
    return a, ACTION_ALIAS[a]


# ============================================================
# Config
# ============================================================
@dataclass
class TrainCfg:
    action_alias: str
    action_internal: str

    backbone_ckpt: str
    data_root: str
    save_root: str
    results_root: str

    epochs: int
    batch_size: int
    num_workers: int
    patch: int
    lr: float
    weight_decay: float

    use_amp: bool
    channels_last: bool
    tf32: bool

    # backbone ctor
    dim: int
    bias: bool
    volterra_rank: int

    lora_rank: int
    lora_alpha: float
    lora_scale_train: float

    iter_save_interval: int
    metric_every: int
    seed: int

    debug_one_step_check: bool
    fail_fast_on_dim_mismatch: bool


def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser()

    ap.add_argument("--action", required=True, help="one of: dedrop,deblur,desnow,derain,dehaze")
    ap.add_argument(
        "--backbone_ckpt",
        default="E:/ReAct-IR/checkpoints/backbone/epoch_021_L0.0204_P31.45_S0.9371.pth",
    )
    ap.add_argument("--data_root", default="E:/ReAct-IR/data")
    ap.add_argument("--save_root", default="E:/ReAct-IR/checkpoints/toolbank_lora")
    ap.add_argument("--results_root", default="E:/ReAct-IR/results/lora_train")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=1)

    # backbone ctor args (important!)
    ap.add_argument("--dim", type=int, default=64, help="VETNet base dim (must match backbone ckpt dim)")
    ap.add_argument("--bias", type=int, default=0, help="1: WithBiasLayerNorm, 0: BiasFreeLayerNorm")
    ap.add_argument("--volterra_rank", type=int, default=2, help="Volterra rank used in backbone")

    ap.add_argument("--lora_rank", type=int, default=2)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--lora_scale_train", type=float, default=1.0)

    ap.add_argument("--iter_save_interval", type=int, default=300)
    ap.add_argument("--metric_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--debug_one_step_check", type=int, default=1)
    ap.add_argument("--fail_fast_on_dim_mismatch", type=int, default=1)

    a = ap.parse_args()
    alias, internal = map_action(a.action)

    return TrainCfg(
        action_alias=alias,
        action_internal=internal,
        backbone_ckpt=a.backbone_ckpt,
        data_root=a.data_root,
        save_root=a.save_root,
        results_root=a.results_root,
        epochs=int(a.epochs),
        batch_size=int(a.batch_size),
        num_workers=int(a.num_workers),
        patch=int(a.patch),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        use_amp=bool(int(a.use_amp)),
        channels_last=bool(int(a.channels_last)),
        tf32=bool(int(a.tf32)),
        dim=int(a.dim),
        bias=bool(int(a.bias)),
        volterra_rank=int(a.volterra_rank),
        lora_rank=int(a.lora_rank),
        lora_alpha=float(a.lora_alpha),
        lora_scale_train=float(a.lora_scale_train),
        iter_save_interval=int(a.iter_save_interval),
        metric_every=int(a.metric_every),
        seed=int(a.seed),
        debug_one_step_check=bool(int(a.debug_one_step_check)),
        fail_fast_on_dim_mismatch=bool(int(a.fail_fast_on_dim_mismatch)),
    )


# ============================================================
# Debug checks
# ============================================================
@torch.no_grad()
def _forward_one(tb: ToolBank, action_internal: str, x: torch.Tensor, amp: bool) -> torch.Tensor:
    tb.activate(action_internal, scale=1.0)
    with autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and x.is_cuda)):
        y = tb(x)
    return y.clamp(0, 1)


def one_step_loss_decrease_check(
    tb: ToolBank,
    action_internal: str,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    use_amp: bool,
) -> Dict[str, float]:
    tb.train()
    tb.activate(action_internal, scale=1.0)

    with torch.no_grad():
        pred0 = _forward_one(tb, action_internal, x, amp=use_amp)
        loss0 = float(F.l1_loss(pred0, y).item())

    opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and x.is_cuda)):
        pred = tb(x)
        loss = F.l1_loss(pred, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

    with torch.no_grad():
        pred1 = _forward_one(tb, action_internal, x, amp=use_amp)
        loss1 = float(F.l1_loss(pred1, y).item())

    return {"loss_before": loss0, "loss_after": loss1, "delta": (loss0 - loss1)}


def save_reload_diff_check(
    cfg: TrainCfg,
    base_model_ctor_kwargs: Dict[str, Any],
    action_internal: str,
    tb: ToolBank,
    x: torch.Tensor,
    use_amp: bool,
    base_ref: torch.nn.Module,
) -> Dict[str, float]:
    """
    LoRA-only fidelity check:
      - y0 = tb(x) with LoRA active
      - save LoRA sd
      - create a fresh base2 loaded from PURE backbone state of base_ref
      - wrap ToolBank tb2, load LoRA sd
      - y1 = tb2(x)
      - diff = max|y0-y1|

    This must be 0 (or tiny fp16 eps) if save/load is correct.
    """
    tb.eval()
    with torch.no_grad():
        y0 = _forward_one(tb, action_internal, x, amp=use_amp)

    lora_sd = lora_state_dict_for_action(tb, action_internal)
    tensors, elems_m = state_dict_stats(lora_sd)

    # Fresh backbone
    base2 = VETNet(**base_model_ctor_kwargs)
    # Load from pure backbone weights extracted from base_ref (not from ckpt file)
    pure_sd = extract_pure_backbone_state_dict(base_ref)
    base2.load_state_dict(pure_sd, strict=False)
    freeze_all(base2)

    tb2 = ToolBank(
        base2,
        actions=["A_DEDROP", "A_DEBLUR", "A_DESNOW", "A_DERAIN", "A_DEHAZE", "A_STOP"],
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        wrap_only_1x1=True,
    )

    device = x.device
    tb2.to(device)
    if cfg.channels_last and device.type == "cuda":
        tb2 = tb2.to(memory_format=torch.channels_last)

    load_lora_state_dict_for_action(tb2, action_internal, lora_sd, strict=True)

    tb2.eval()
    with torch.no_grad():
        y1 = _forward_one(tb2, action_internal, x, amp=use_amp)

    diff = float((y0 - y1).abs().max().item())
    return {"state_tensors": float(tensors), "state_elems_M": float(elems_m), "max_abs_diff_reload": diff}


# ============================================================
# Main train
# ============================================================
def main():
    cfg = parse_args()
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[SKIMAGE]", "ON" if USE_SKIMAGE else "OFF")

    torch.backends.cudnn.benchmark = True
    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"[Speed] tf32={cfg.tf32} channels_last={cfg.channels_last} amp={cfg.use_amp}")

    print("\n[Action] alias =", cfg.action_alias, "-> internal =", cfg.action_internal)

    # ---------------- dataset ----------------
    pair_cfg = ActionPairConfig(action=cfg.action_internal, split="train", data_root=cfg.data_root)
    pairs, _reports = build_action_pairs(pair_cfg)

    by_ds: Dict[str, int] = {}
    for _, _, m in pairs:
        by_ds[m["dataset"]] = by_ds.get(m["dataset"], 0) + 1
    print("[Pairs] total =", len(pairs))
    for k in sorted(by_ds.keys()):
        print(f"  - {k}: {by_ds[k]}")

    ds = LoRAPairedDataset(pairs=pairs, patch=cfg.patch, augment=True, strict_size_check=False)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_lora,
        persistent_workers=(cfg.num_workers > 0),
    )
    steps_per_epoch = len(loader)
    print(f"[Train] steps_per_epoch={steps_per_epoch} batch={cfg.batch_size}")

    # ---------------- model + toolbank ----------------
    base_ctor = dict(dim=cfg.dim, bias=cfg.bias, volterra_rank=cfg.volterra_rank)
    base = VETNet(**base_ctor)

    # robust compat load (also does ckpt dim check)
    print(f"[Backbone] ctor: dim={cfg.dim} bias={cfg.bias} volterra_rank={cfg.volterra_rank}")
    _ckpt_info = load_backbone_ckpt_compat(
        base,
        cfg.backbone_ckpt,
        expect_dim=cfg.dim,
        fail_fast_on_dim_mismatch=cfg.fail_fast_on_dim_mismatch,
        verbose=True,
    )

    # freeze backbone
    freeze_all(base)

    tb = ToolBank(
        base,
        actions=["A_DEDROP", "A_DEBLUR", "A_DESNOW", "A_DERAIN", "A_DEHAZE", "A_STOP"],
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        wrap_only_1x1=True,
    )

    tb.to(device)
    if cfg.channels_last and device.type == "cuda":
        tb = tb.to(memory_format=torch.channels_last)

    if hasattr(tb, "summary"):
        s = tb.summary()
        print(f"[ToolBank] injected LoRA into {s.get('wrapped_layers','?')} convs (expect ~179)")
    else:
        print("[ToolBank] injected (summary() not available)")

    # Make sure everything is frozen before enabling single action
    freeze_all(tb)
    tp0 = count_trainable_params(tb)
    print(f"[DEBUG] trainable_params(after freeze_all(tb))={tp0:.6f}M (expect 0.0M)")

    tb.set_trainable_action(cfg.action_internal)
    tp1 = count_trainable_params(tb)
    print(f"[DEBUG] trainable_params(after set_trainable_action)={tp1:.6f}M")

    trainable = [p for p in tb.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable params found after set_trainable_action().")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    save_dir = os.path.join(cfg.save_root, cfg.action_alias)
    res_dir = os.path.join(cfg.results_root, cfg.action_alias)
    safe_makedirs(save_dir)
    safe_makedirs(res_dir)
    safe_makedirs(os.path.join(res_dir, "iter"))

    # ---------------- debug checks: one batch ----------------
    x_dbg, y_dbg, _meta_dbg = next(iter(loader))
    if cfg.channels_last and device.type == "cuda":
        x_dbg = x_dbg.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y_dbg = y_dbg.to(device, non_blocking=True).to(memory_format=torch.channels_last)
    else:
        x_dbg = x_dbg.to(device, non_blocking=True)
        y_dbg = y_dbg.to(device, non_blocking=True)

    if cfg.debug_one_step_check:
        print("\n[DEBUG] one-step loss decrease check (same batch)")
        out = one_step_loss_decrease_check(tb, cfg.action_internal, opt, scaler, x_dbg, y_dbg, cfg.use_amp)
        print(
            f"  loss_before={out['loss_before']:.6f}  "
            f"loss_after={out['loss_after']:.6f}  "
            f"delta={out['delta']:.6f}"
        )
        print("  decreased =", (out["loss_after"] <= out["loss_before"]))

        print("\n[DEBUG] save->reload diff check (same input)  [LoRA-only fidelity]")
        out2 = save_reload_diff_check(cfg, base_ctor, cfg.action_internal, tb, x_dbg[:1], cfg.use_amp, base)
        print(
            f"  saved tensors={int(out2['state_tensors'])} "
            f"elems={out2['state_elems_M']:.6f}M "
            f"max_abs_diff_reload={out2['max_abs_diff_reload']:.8f}"
        )

    # ---------------- training loop ----------------
    print("\n[Train] start")
    for epoch in range(1, cfg.epochs + 1):
        tb.train()
        tb.activate(cfg.action_internal, scale=cfg.lora_scale_train)

        loss_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        metric_cnt = 0

        t0 = time.time()
        pbar = tqdm(loader, ncols=140, desc=f"Epoch {epoch:03d}/{cfg.epochs}")

        for it, (inp, gt, _metas) in enumerate(pbar, start=1):
            if cfg.channels_last and device.type == "cuda":
                inp = inp.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                gt = gt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                inp = inp.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                pred = tb(inp)
                loss = F.l1_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item())
            pred_c = pred.detach().clamp(0, 1)

            if cfg.iter_save_interval > 0 and (it % cfg.iter_save_interval == 0):
                outp = os.path.join(res_dir, "iter", f"epoch_{epoch:03d}_iter_{it:05d}.png")
                save_triplet(inp[0].detach().cpu(), pred_c[0].cpu(), gt[0].detach().cpu(), outp)

            if USE_SKIMAGE and cfg.metric_every > 0 and (it % cfg.metric_every == 0 or it == steps_per_epoch):
                ps, ss = compute_psnr_ssim_1(pred_c[0].cpu(), gt[0].cpu())
                psnr_sum += ps
                ssim_sum += ss
                metric_cnt += 1

            elapsed = time.time() - t0
            it_per_sec = it / max(elapsed, 1e-6)
            eta_sec = (steps_per_epoch - it) / max(it_per_sec, 1e-6)

            avg_loss = loss_sum / it
            avg_psnr = (psnr_sum / metric_cnt) if metric_cnt > 0 else 0.0
            avg_ssim = (ssim_sum / metric_cnt) if metric_cnt > 0 else 0.0

            pbar.set_postfix(
                {
                    "L": f"{avg_loss:.4f}",
                    "P": f"{avg_psnr:.2f}" if USE_SKIMAGE else "NA",
                    "S": f"{avg_ssim:.3f}" if USE_SKIMAGE else "NA",
                    "ETA": format_time(eta_sec),
                    "lr": f"{cfg.lr:.1e}",
                }
            )

        epoch_loss = loss_sum / max(steps_per_epoch, 1)
        epoch_psnr = (psnr_sum / metric_cnt) if metric_cnt > 0 else 0.0
        epoch_ssim = (ssim_sum / metric_cnt) if metric_cnt > 0 else 0.0
        print(
            f"\n[Epoch {epoch:03d}] Loss={epoch_loss:.6f}  "
            f"PSNR={epoch_psnr:.2f}  SSIM={epoch_ssim:.4f}  time={format_time(time.time()-t0)}"
        )

        # ---------------- save LoRA only ----------------
        sd = lora_state_dict_for_action(tb, cfg.action_internal)
        tensors, elems_m = state_dict_stats(sd)

        ckpt = {
            "action_alias": cfg.action_alias,
            "action_internal": cfg.action_internal,
            "epoch": epoch,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_state_dict": sd,
            "metrics": {
                "loss": float(epoch_loss),
                "psnr": float(epoch_psnr),
                "ssim": float(epoch_ssim),
                "metric_cnt": int(metric_cnt),
                "use_skimage": bool(USE_SKIMAGE),
            },
            "cfg": vars(cfg),
        }

        ckpt_name = f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth"
        ckpt_path = os.path.join(save_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)
        print(f"[Save] {ckpt_path}  (tensors={tensors} elems={elems_m:.6f}M)")

    print("\n[Train] finished")


if __name__ == "__main__":
    main()


"""
python -u e:/ReAct-IR/scripts/train_lora_action.py `
  --action desnow `
  --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/best_backbone.pth" `
  --data_root "E:/ReAct-IR/data" `
  --save_root "E:/ReAct-IR/checkpoints/toolbank_lora_dim64" `
  --results_root "E:/ReAct-IR/results/lora_train_dim64" `
  --epochs 20 --batch_size 1 --patch 256 --lr 3e-4 `
  --dim 64 --bias 0 --volterra_rank 2 `
  --lora_rank 2 --lora_alpha 1.0 --use_amp 1 `
  --iter_save_interval 300 --metric_every 200

"""
