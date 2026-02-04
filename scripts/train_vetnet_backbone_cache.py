# e:/ReAct-IR/scripts/train_vetnet_backbone_cache.py
# ------------------------------------------------------------
# Train VETNet backbone on preload_cache (paired *_in / *_gt png/jpg/tif)
# - Scans: E:/ReAct-IR/preload_cache/{CSD, DayRainDrop, NightRainDrop, rain100H, RESIDE-6K}
# - Ignores: *_clip.pt and any non-image files
# - AMP + tqdm with loss/psnr/ssim/eta
# - PSNR/SSIM computed only every --metric_every steps
# - Speed knobs: TF32, channels_last, grad_accum, optional torch.compile
#
# NEW (2026-02-04): WeightedRandomSampler oversampling WITHOUT increasing epoch length
#   - raindrop (DayRainDrop/NightRainDrop): 6x frequency
#   - rain100H: 2x frequency
#   - CSD / RESIDE-6K: 1x
#   => steps_per_epoch unchanged, but sampling distribution adjusted
#
# Example (Windows PowerShell):
#   python e:/ReAct-IR/scripts/train_vetnet_backbone_cache.py `
#     --cache_root E:/ReAct-IR/preload_cache `
#     --datasets CSD,DayRainDrop,NightRainDrop,rain100H,RESIDE-6K `
#     --epochs 100 --batch_size 1 --grad_accum 2 --patch 256 --lr 3e-4 `
#     --dim 64 --bias 0 --volterra_rank 2 `
#     --num_workers 0 --metric_every 200 --iter_save_interval 300
# ------------------------------------------------------------

import os
import sys
import glob
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# ------------------------------------------------------------
# Make project import-safe
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet  # uses models/backbone/{mdta,gdfn,volterra}.py

# ------------------------------------------------------------
# Optional skimage metrics (slow -> use sparingly)
# ------------------------------------------------------------
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False

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
    return np.array(Image.open(path).convert("RGB"))


def to_chw_float01(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1)


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
    canvas[:, w : 2 * w] = pr
    canvas[:, 2 * w : 3 * w] = gt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)


def compute_psnr_ssim_batch(pred: torch.Tensor, gt: torch.Tensor, max_images: int = 1) -> Tuple[float, float]:
    """
    pred, gt: BCHW float in [0,1]
    Computes on first max_images samples (CPU, slow).
    """
    if not USE_SKIMAGE:
        return 0.0, 0.0
    b = int(pred.shape[0])
    n = min(b, max_images)
    ps_sum = 0.0
    ss_sum = 0.0
    for i in range(n):
        p = tensor_to_img_u8(pred[i])
        g = tensor_to_img_u8(gt[i])
        ps_sum += float(peak_signal_noise_ratio(g, p, data_range=255))
        ss_sum += float(structural_similarity(g, p, channel_axis=2, data_range=255))
    return ps_sum / n, ss_sum / n


def format_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec - 60 * m)
    h = int(m // 60)
    m = int(m - 60 * h)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def count_by_dataset(pairs: List[Tuple[str, str, Dict]]) -> Dict[str, int]:
    by_ds: Dict[str, int] = {}
    for _, _, m in pairs:
        dn = m.get("dataset", "UNKNOWN")
        by_ds[dn] = by_ds.get(dn, 0) + 1
    return by_ds


# ============================================================
# Cache pairing
# ============================================================
def build_pairs_from_cache(cache_root: str, subfolders: List[str]) -> List[Tuple[str, str, Dict]]:
    pairs: List[Tuple[str, str, Dict]] = []

    for name in subfolders:
        droot = os.path.join(cache_root, name)
        if not os.path.isdir(droot):
            continue

        in_files = []
        for ext in IMG_EXTS:
            in_files += glob.glob(os.path.join(droot, f"*_in{ext}"))
        in_files = sorted(in_files)

        gt_map: Dict[str, str] = {}
        gt_files = []
        for ext in IMG_EXTS:
            gt_files += glob.glob(os.path.join(droot, f"*_gt{ext}"))
        for g in gt_files:
            base = os.path.basename(g)
            if "_gt" not in base:
                continue
            key = base.split("_gt")[0]
            gt_map[key] = g

        for ip in in_files:
            base = os.path.basename(ip)
            if "_in" not in base:
                continue
            key = base.split("_in")[0]
            gp = gt_map.get(key, None)
            if gp is None:
                continue
            meta = {"dataset": name, "id": key, "input_path": ip, "gt_path": gp}
            pairs.append((ip, gp, meta))

    return pairs


# ============================================================
# Dataset
# ============================================================
class PairedCacheDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, Dict]], patch: int = 256, augment: bool = True):
        self.pairs = pairs
        self.patch = int(patch)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
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
        return inp[y0 : y0 + patch, x0 : x0 + patch], gt[y0 : y0 + patch, x0 : x0 + patch]

    @staticmethod
    def _augment_pair(inp: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            inp = inp[:, ::-1].copy()
            gt = gt[:, ::-1].copy()
        if random.random() < 0.5:
            inp = inp[::-1, :, :].copy()
            gt = gt[::-1, :, :].copy()
        k = random.randint(0, 3)
        if k:
            inp = np.rot90(inp, k, axes=(0, 1)).copy()
            gt = np.rot90(gt, k, axes=(0, 1)).copy()
        return inp, gt

    def __getitem__(self, idx: int) -> Dict:
        inp_path, gt_path, meta = self.pairs[idx]
        inp = imread_rgb(inp_path)
        gt = imread_rgb(gt_path)
        inp, gt = self._random_crop_pair(inp, gt, self.patch)
        if self.augment:
            inp, gt = self._augment_pair(inp, gt)
        inp_t = to_chw_float01(inp)
        gt_t = to_chw_float01(gt)
        return {"input": inp_t, "gt": gt_t, "meta": meta}


def collate_dict(batch: List[Dict]):
    xs = torch.stack([b["input"] for b in batch], dim=0)
    ys = torch.stack([b["gt"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return xs, ys, metas


# ============================================================
# Train
# ============================================================
@dataclass
class TrainConfig:
    cache_root: str
    datasets: List[str]
    save_root: str
    results_root: str

    epochs: int
    batch_size: int
    grad_accum: int
    num_workers: int
    lr: float
    patch: int

    dim: int
    bias: bool
    volterra_rank: int

    use_amp: bool
    compile: bool
    channels_last: bool
    tf32: bool

    metric_every: int
    metric_images_per_batch: int
    iter_save_interval: int
    preview_count: int

    # sampler
    use_weighted_sampler: bool
    w_dayraindrop: float
    w_nightraindrop: float
    w_rain100h: float

    seed: int


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache_root", default="E:/ReAct-IR/preload_cache")
    ap.add_argument("--datasets", default="CSD,DayRainDrop,NightRainDrop,rain100H,RESIDE-6K")

    ap.add_argument("--save_root", default="E:/ReAct-IR/checkpoints/backbone")
    ap.add_argument("--results_root", default="E:/ReAct-IR/results/backbone_train")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patch", type=int, default=256)

    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--bias", type=int, default=1)  # 0/1
    ap.add_argument("--volterra_rank", type=int, default=2)

    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=1)

    ap.add_argument("--metric_every", type=int, default=200)
    ap.add_argument("--metric_images_per_batch", type=int, default=1)
    ap.add_argument("--iter_save_interval", type=int, default=300)
    ap.add_argument("--preview_count", type=int, default=3)

    # sampler knobs (epoch length fixed)
    ap.add_argument("--use_weighted_sampler", type=int, default=1)
    ap.add_argument("--w_dayraindrop", type=float, default=6.0)
    ap.add_argument("--w_nightraindrop", type=float, default=6.0)
    ap.add_argument("--w_rain100h", type=float, default=2.0)

    ap.add_argument("--seed", type=int, default=123)

    a = ap.parse_args()

    return TrainConfig(
        cache_root=a.cache_root,
        datasets=[s.strip() for s in a.datasets.split(",") if s.strip()],
        save_root=a.save_root,
        results_root=a.results_root,
        epochs=int(a.epochs),
        batch_size=int(a.batch_size),
        grad_accum=max(1, int(a.grad_accum)),
        num_workers=int(a.num_workers),
        lr=float(a.lr),
        patch=int(a.patch),
        dim=int(a.dim),
        bias=bool(int(a.bias)),
        volterra_rank=int(a.volterra_rank),
        use_amp=bool(int(a.use_amp)),
        compile=bool(int(a.compile)),
        channels_last=bool(int(a.channels_last)),
        tf32=bool(int(a.tf32)),
        metric_every=int(a.metric_every),
        metric_images_per_batch=int(a.metric_images_per_batch),
        iter_save_interval=int(a.iter_save_interval),
        preview_count=int(a.preview_count),
        use_weighted_sampler=bool(int(a.use_weighted_sampler)),
        w_dayraindrop=float(a.w_dayraindrop),
        w_nightraindrop=float(a.w_nightraindrop),
        w_rain100h=float(a.w_rain100h),
        seed=int(a.seed),
    )


def main():
    cfg = parse_args()
    seed_all(cfg.seed)

    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.results_root, exist_ok=True)
    os.makedirs(os.path.join(cfg.results_root, "iter"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[SKIMAGE]", "ON" if USE_SKIMAGE else "OFF")

    # speed knobs
    torch.backends.cudnn.benchmark = True
    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"[Speed] tf32={cfg.tf32} channels_last={cfg.channels_last} compile={cfg.compile} grad_accum={cfg.grad_accum}")

    pairs = build_pairs_from_cache(cfg.cache_root, cfg.datasets)
    if len(pairs) == 0:
        raise RuntimeError(f"No pairs found. cache_root={cfg.cache_root} datasets={cfg.datasets}")

    by_ds_raw = count_by_dataset(pairs)
    print("[CachePairs] total =", len(pairs))
    for k in sorted(by_ds_raw.keys()):
        print(f"  - {k}: {by_ds_raw[k]}")

    ds = PairedCacheDataset(pairs=pairs, patch=cfg.patch, augment=True)

    # ------------------------------------------------------------
    # WeightedRandomSampler: change frequency WITHOUT increasing epoch length
    # ------------------------------------------------------------
    sampler = None
    if cfg.use_weighted_sampler:
        W = {
            "DayRainDrop": float(cfg.w_dayraindrop),
            "NightRainDrop": float(cfg.w_nightraindrop),
            "rain100H": float(cfg.w_rain100h),
            # CSD / RESIDE-6K default 1.0
        }
        weights = [float(W.get(m.get("dataset", ""), 1.0)) for _, _, m in pairs]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(pairs),  # ✅ epoch length fixed
            replacement=True,
        )
        print(
            "[Sampler] WeightedRandomSampler ON "
            f"| num_samples={len(pairs)} "
            f"| w_dayraindrop={cfg.w_dayraindrop} w_nightraindrop={cfg.w_nightraindrop} w_rain100H={cfg.w_rain100h}"
        )
    else:
        print("[Sampler] OFF (plain shuffle)")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_dict,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model = VETNet(dim=cfg.dim, bias=cfg.bias, volterra_rank=cfg.volterra_rank).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] VETNet dim={cfg.dim} bias={cfg.bias} volterra_rank={cfg.volterra_rank} | params={nparams:.2f}M")

    # optional compile (can be slower on some setups; keep as opt-in)
    if cfg.compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[Compile] torch.compile ON")
        except Exception as e:
            print("[Compile] failed -> OFF:", repr(e))

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    steps_per_epoch = len(loader)
    print(f"[Train] steps_per_epoch={steps_per_epoch} (pairs={len(ds)} batch={cfg.batch_size} accum={cfg.grad_accum})")

    best_ssim = -1.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        loss_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        metric_cnt = 0

        t0 = time.time()

        preview_inp = None
        preview_gt = None
        preview_pred = None

        pbar = tqdm(loader, ncols=140, desc=f"Epoch {epoch:03d}/{cfg.epochs}")

        optim.zero_grad(set_to_none=True)

        for it, (inp, gt, metas) in enumerate(pbar, start=1):
            global_step += 1

            if cfg.channels_last and device.type == "cuda":
                inp = inp.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                gt = gt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                inp = inp.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
                pred = model(inp)
                loss = F.l1_loss(pred, gt) / float(cfg.grad_accum)

            scaler.scale(loss).backward()

            # step on accumulation boundary
            if (it % cfg.grad_accum) == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            loss_sum += float(loss.item()) * float(cfg.grad_accum)

            pred_c = pred.clamp(0, 1)

            if preview_inp is None:
                preview_inp = inp.detach().cpu()
                preview_gt = gt.detach().cpu()
                preview_pred = pred_c.detach().cpu()

            if cfg.iter_save_interval > 0 and (it % cfg.iter_save_interval == 0):
                outp = os.path.join(cfg.results_root, "iter", f"epoch_{epoch:03d}_iter_{it:05d}.png")
                save_triplet(inp[0].detach().cpu(), pred_c[0].detach().cpu(), gt[0].detach().cpu(), outp)

            if USE_SKIMAGE and cfg.metric_every > 0 and (it % cfg.metric_every == 0 or it == steps_per_epoch):
                ps, ss = compute_psnr_ssim_batch(pred_c, gt, max_images=cfg.metric_images_per_batch)
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
                    "lr": f"{optim.param_groups[0]['lr']:.1e}",
                }
            )

        # if steps_per_epoch not divisible by grad_accum, flush remaining grads
        if (steps_per_epoch % cfg.grad_accum) != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        sched.step()

        epoch_loss = loss_sum / max(steps_per_epoch, 1)
        epoch_psnr = (psnr_sum / metric_cnt) if metric_cnt > 0 else 0.0
        epoch_ssim = (ssim_sum / metric_cnt) if metric_cnt > 0 else 0.0

        print(f"\n[Epoch {epoch:03d}] Loss={epoch_loss:.4f}  PSNR={epoch_psnr:.2f}  SSIM={epoch_ssim:.4f}  time={format_time(time.time()-t0)}")

        # preview
        if preview_inp is not None:
            total = preview_inp.size(0)
            k = min(cfg.preview_count, total)
            idxs = np.random.choice(total, k, replace=False)
            for j, idx in enumerate(idxs):
                p = os.path.join(cfg.results_root, f"epoch_{epoch:03d}_preview_{j:02d}.png")
                save_triplet(preview_inp[idx], preview_pred[idx], preview_gt[idx], p)

            named = os.path.join(cfg.results_root, f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.png")
            save_triplet(preview_inp[0], preview_pred[0], preview_gt[0], named)

        ckpt_path = os.path.join(cfg.save_root, f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "cfg": vars(cfg),
            },
            ckpt_path,
        )

        if USE_SKIMAGE and epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            best_path = os.path.join(cfg.save_root, "best_backbone.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] Updated best SSIM={best_ssim:.4f} -> {best_path}")

    print("\nTraining finished.")
    if USE_SKIMAGE:
        print("Best SSIM:", best_ssim)


if __name__ == "__main__":
    main()


