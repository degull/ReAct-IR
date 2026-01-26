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
#   E:/ReAct-IR/checkpoints/toolbank_lora/{action}/epoch_XXX.pth
# where the checkpoint contains ONLY LoRA weights for that action.
#
# Also saves preview images (inp|pred|gt) every --iter_save_interval.
#
# Includes __main__ debug checks:
# - 1 step loss(before/after) on same batch
# - saved state_dict tensor/elem counts
# - save->reload output diff check
# ------------------------------------------------------------

import os
import sys
import time
import math
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
    canvas[:, w:2 * w] = pr
    canvas[:, 2 * w:3 * w] = gt

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


def load_backbone_ckpt(model: torch.nn.Module, ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("patch_embed") or k.startswith("encoder") for k in ckpt.keys()):
        sd = ckpt
    else:
        # fallback: maybe saved as plain state_dict
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    info = {
        "ckpt_path": ckpt_path,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "example_missing": missing[:10],
        "example_unexpected": unexpected[:10],
        "has_state_dict": isinstance(ckpt, dict) and "state_dict" in ckpt,
    }
    return info


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def lora_state_dict_for_action(tb: ToolBank, action: str) -> Dict[str, torch.Tensor]:
    # support either API
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


def state_dict_stats(sd: Dict[str, torch.Tensor]) -> Tuple[int, float]:
    tensors = 0
    elems = 0
    for _, v in sd.items():
        if torch.is_tensor(v):
            tensors += 1
            elems += int(v.numel())
    return tensors, elems / 1e6


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

    lora_rank: int
    lora_alpha: float
    lora_scale_train: float

    iter_save_interval: int
    metric_every: int
    seed: int

    debug_one_step_check: bool


def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser()

    ap.add_argument("--action", required=True, help="one of: dedrop,deblur,desnow,derain,dehaze")
    ap.add_argument("--backbone_ckpt", default="E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth")
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

    ap.add_argument("--lora_rank", type=int, default=2)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--lora_scale_train", type=float, default=1.0)

    ap.add_argument("--iter_save_interval", type=int, default=300)
    ap.add_argument("--metric_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--debug_one_step_check", type=int, default=1)

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
        lora_rank=int(a.lora_rank),
        lora_alpha=float(a.lora_alpha),
        lora_scale_train=float(a.lora_scale_train),
        iter_save_interval=int(a.iter_save_interval),
        metric_every=int(a.metric_every),
        seed=int(a.seed),
        debug_one_step_check=bool(int(a.debug_one_step_check)),
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
) -> Dict[str, float]:
    tb.eval()

    with torch.no_grad():
        y0 = _forward_one(tb, action_internal, x, amp=use_amp)

    sd = lora_state_dict_for_action(tb, action_internal)
    tensors, elems_m = state_dict_stats(sd)

    base2 = VETNet(**base_model_ctor_kwargs)
    _ = load_backbone_ckpt(base2, cfg.backbone_ckpt)
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

    load_lora_state_dict_for_action(tb2, action_internal, sd, strict=True)

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
    pairs, reports = build_action_pairs(pair_cfg)

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
    base_ctor = dict(dim=48, bias=False, volterra_rank=4)
    base = VETNet(**base_ctor)
    ckpt_info = load_backbone_ckpt(base, cfg.backbone_ckpt)
    print("[Backbone] ckpt =", ckpt_info["ckpt_path"])
    print(f"[Backbone] loaded (missing={ckpt_info['missing_keys']} unexpected={ckpt_info['unexpected_keys']})")
    if ckpt_info["missing_keys"] or ckpt_info["unexpected_keys"]:
        print("  example_missing   :", ckpt_info["example_missing"])
        print("  example_unexpected:", ckpt_info["example_unexpected"])

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

    tp0 = count_trainable_params(tb)
    print(f"[DEBUG] trainable_params(before set_trainable_action)={tp0:.6f}M (expect 0.0M)")

    tb.set_trainable_action(cfg.action_internal)
    tp1 = count_trainable_params(tb)
    print(f"[DEBUG] trainable_params(after set_trainable_action) ={tp1:.6f}M (expect ~0.23M)")

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
    x_dbg, y_dbg, meta_dbg = next(iter(loader))
    if cfg.channels_last and device.type == "cuda":
        x_dbg = x_dbg.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y_dbg = y_dbg.to(device, non_blocking=True).to(memory_format=torch.channels_last)
    else:
        x_dbg = x_dbg.to(device, non_blocking=True)
        y_dbg = y_dbg.to(device, non_blocking=True)

    if cfg.debug_one_step_check:
        print("\n[DEBUG] one-step loss decrease check (same batch)")
        out = one_step_loss_decrease_check(tb, cfg.action_internal, opt, scaler, x_dbg, y_dbg, cfg.use_amp)
        print(f"  loss_before={out['loss_before']:.6f}  loss_after={out['loss_after']:.6f}  delta={out['delta']:.6f}")
        print("  decreased =", (out["loss_after"] <= out["loss_before"]))

        print("\n[DEBUG] save->reload diff check (same input)")
        out2 = save_reload_diff_check(cfg, base_ctor, cfg.action_internal, tb, x_dbg[:1], cfg.use_amp)
        print(f"  saved tensors={int(out2['state_tensors'])} elems={out2['state_elems_M']:.6f}M max_abs_diff_reload={out2['max_abs_diff_reload']:.8f}")

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

        for it, (inp, gt, metas) in enumerate(pbar, start=1):
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

            pbar.set_postfix({
                "L": f"{avg_loss:.4f}",
                "P": f"{avg_psnr:.2f}" if USE_SKIMAGE else "NA",
                "S": f"{avg_ssim:.3f}" if USE_SKIMAGE else "NA",
                "ETA": format_time(eta_sec),
                "lr": f"{cfg.lr:.1e}",
            })

        epoch_loss = loss_sum / max(steps_per_epoch, 1)
        epoch_psnr = (psnr_sum / metric_cnt) if metric_cnt > 0 else 0.0
        epoch_ssim = (ssim_sum / metric_cnt) if metric_cnt > 0 else 0.0
        print(f"\n[Epoch {epoch:03d}] Loss={epoch_loss:.6f}  PSNR={epoch_psnr:.2f}  SSIM={epoch_ssim:.4f}  time={format_time(time.time()-t0)}")

        # ---------------- save LoRA only (filename: epoch_004_L0.0367_P27.24_S0.8868.pth) ----------------
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
Examples (PowerShell):

# DESNOW
python -u e:/ReAct-IR/scripts/train_lora_action.py `
   --action desnow `
   --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
   --data_root "E:/ReAct-IR/data" `
   --save_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
   --results_root "E:/ReAct-IR/results/lora_train" `
   --epochs 20 --batch_size 1 --patch 256 --lr 3e-4 `
   --lora_rank 2 --lora_alpha 1.0 --use_amp 1 `
   --iter_save_interval 300 --metric_every 200
 
# DEDROP
python -u e:/ReAct-IR/scripts/train_lora_action.py `
   --action dedrop `
   --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
   --data_root "E:/ReAct-IR/data" `
   --save_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
   --results_root "E:/ReAct-IR/results/lora_train" `
   --epochs 20 --batch_size 1 --patch 256 --lr 3e-4 `
   --lora_rank 2 --lora_alpha 1.0 --use_amp 1 `
   --iter_save_interval 300 --metric_every 200

# DERAIN
python -u e:/ReAct-IR/scripts/train_lora_action.py `
   --action derain `
   --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
   --data_root "E:/ReAct-IR/data" `
   --save_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
   --results_root "E:/ReAct-IR/results/lora_train" `
   --epochs 20 --batxwcize 1 --patch 256 --lr 3e-4 `
   --lora_rank 2 --lora_alpha 1.0 --use_amp 1 `
   --iter_save_interval 300 --metric_every 200


# DEHAZE
python -u e:/ReAct-IR/scripts/train_lora_action.py `
   --action dehaze `
   --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
   --data_root "E:/ReAct-IR/data" `
   --save_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
   --results_root "E:/ReAct-IR/results/lora_train" `
   --epochs 20 --batch_size 1 --patch 256 --lr 3e-4 `
   --lora_rank 2 --lora_alpha 1.0 --use_amp 1 `
   --iter_save_interval 300 --metric_every 200


# DEBLUR
python -u e:/ReAct-IR/scripts/train_lora_action.py `
   --action deblur `
   --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
   --data_root "E:/ReAct-IR/data" `
   --save_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
   --results_root "E:/ReAct-IR/results/lora_train" `
   --epochs 20 --batch_size 1 --patch 256 --lr 3e-4 `
   --lora_rank 2 --lora_alpha 1.0 --use_amp 1 `
   --iter_save_interval 300 --metric_every 200
"""