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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.lora_dataset import (
    ActionPairConfig,
    build_action_pairs,
    LoRAPairedDataset,
)

# ============================================================
# Labels / mapping
# ============================================================
# label order: [blur, rain, snow, haze, drop]
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
ACTION_TO_LABEL_INDEX = {
    "A_DEBLUR": 0,
    "A_DERAIN": 1,
    "A_DESNOW": 2,
    "A_DEHAZE": 3,
    "A_DEDROP": 4,
}
ACTIONS_INTERNAL = list(ACTION_TO_LABEL_INDEX.keys())

SNOW_CH = ACTION_TO_LABEL_INDEX["A_DESNOW"]  # 2


# ============================================================
# Utils
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def format_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec - 60 * m)
    h = int(m // 60)
    m = int(m - 60 * h)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _pil_load_mask_as_01(path: str) -> torch.Tensor:
    """
    Load mask as float tensor in [0,1], shape (H,W).
    Supports common mask formats (0/255, etc.).
    """
    im = Image.open(path).convert("L")
    arr = np.array(im).astype(np.float32)
    # normalize heuristics
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr)  # (H,W)


def infer_csd_mask_path(inp_path: str) -> Optional[str]:
    """
    Robustly infer CSD mask path from an input path.
    CSD structure you mentioned: .../CSD/{Train|Test}/{Snow,Mask,Gt}/xxx.tif
    """
    p = inp_path.replace("\\", "/")
    if "/CSD/" not in p:
        return None
    if "/Snow/" not in p and "/snow/" not in p:
        return None

    # replace /Snow/ with /Mask/
    p2 = p.replace("/Snow/", "/Mask/").replace("/snow/", "/Mask/")
    # try same filename
    cand = p2
    if os.path.exists(cand):
        return cand

    # sometimes extension differs, try common ones
    base, ext = os.path.splitext(cand)
    for e in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        c = base + e
        if os.path.exists(c):
            return c
    return None


# ============================================================
# Dataset: mixed action dataset, returns (inp, gt, label5, mask_or_none, meta)
# ============================================================
class DiagnoserMapDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        patch: int,
        augment: bool,
        max_per_action: int = -1,
        seed: int = 123,
        strict_size_check: bool = False,
        use_csd_mask: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.patch = patch
        self.augment = augment
        self.max_per_action = int(max_per_action)
        self.seed = int(seed)
        self.strict_size_check = strict_size_check
        self.use_csd_mask = bool(use_csd_mask)

        rng = random.Random(self.seed)

        all_pairs: List[Tuple[str, str, Dict[str, Any]]] = []
        self.stats_by_action: Dict[str, int] = {}

        for action in ACTIONS_INTERNAL:
            pair_cfg = ActionPairConfig(action=action, split=split, data_root=data_root)
            pairs, reports = build_action_pairs(pair_cfg)

            if self.max_per_action > 0 and len(pairs) > self.max_per_action:
                pairs = pairs.copy()
                rng.shuffle(pairs)
                pairs = pairs[: self.max_per_action]

            new_pairs = []
            for inp_path, gt_path, meta in pairs:
                meta2 = dict(meta)
                meta2["action_internal"] = action
                meta2["inp_path"] = inp_path
                meta2["gt_path"] = gt_path
                # keep dataset name if exists
                # meta2["dataset"] expected in your codebase
                new_pairs.append((inp_path, gt_path, meta2))

            all_pairs.extend(new_pairs)
            self.stats_by_action[action] = len(new_pairs)

        rng.shuffle(all_pairs)

        # reuse your paired dataset loader for cropping/augment
        self.base = LoRAPairedDataset(
            pairs=all_pairs,
            patch=self.patch,
            augment=self.augment,
            strict_size_check=self.strict_size_check,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        inp, gt, meta = self.base[idx]
        action = meta.get("action_internal")
        if action is None:
            raise RuntimeError("meta must contain 'action_internal'")

        # global one-hot label
        y = torch.zeros(5, dtype=torch.float32)
        y[ACTION_TO_LABEL_INDEX[action]] = 1.0

        # optional CSD snow mask (only meaningful for A_DESNOW)
        mask = None
        if self.use_csd_mask and action == "A_DESNOW":
            inp_path = meta.get("inp_path", None)
            # try meta hints first
            mpath = meta.get("mask_path", None) or meta.get("mask", None)
            if mpath is None and isinstance(inp_path, str):
                mpath = infer_csd_mask_path(inp_path)
            if mpath is not None and os.path.exists(mpath):
                mask_hw = _pil_load_mask_as_01(mpath)  # (H,W) original
                # IMPORTANT: base dataset likely returns cropped patches; we need mask aligned to crop
                # If LoRAPairedDataset stores crop info in meta (x0,y0,...), use it; otherwise fallback to center-crop.
                x0 = meta.get("crop_x", None)
                y0 = meta.get("crop_y", None)
                ph = inp.shape[1]  # H
                pw = inp.shape[2]  # W
                if x0 is not None and y0 is not None:
                    x0 = int(x0); y0 = int(y0)
                    mask_hw = mask_hw[y0:y0+ph, x0:x0+pw]
                else:
                    # fallback: center crop (best-effort)
                    Hm, Wm = mask_hw.shape
                    cy = max(0, (Hm - ph) // 2)
                    cx = max(0, (Wm - pw) // 2)
                    mask_hw = mask_hw[cy:cy+ph, cx:cx+pw]

                # ensure size matches inp patch
                if mask_hw.shape[0] != ph or mask_hw.shape[1] != pw:
                    # resize to patch size
                    mask_hw = torch.from_numpy(
                        np.array(Image.fromarray((mask_hw.numpy()*255).astype(np.uint8)).resize((pw, ph), resample=Image.NEAREST)).astype(np.float32) / 255.0
                    )

                mask = mask_hw.unsqueeze(0)  # (1,H,W) snow mask only

        return inp, gt, y, mask, meta


def collate_diagnoser_map(batch):
    inps = torch.stack([b[0] for b in batch], dim=0)   # (B,3,H,W)
    gts  = torch.stack([b[1] for b in batch], dim=0)   # (B,3,H,W)
    ys   = torch.stack([b[2] for b in batch], dim=0)   # (B,5)
    masks = [b[3] for b in batch]                      # list of None or (1,H,W)
    metas = [b[4] for b in batch]
    return inps, gts, ys, masks, metas


# ============================================================
# Model: ViT-like encoder that outputs both s (global) and m (map)
# ============================================================
class TinyViTDiagnoserWithMap(nn.Module):
    """
    Outputs:
      s_logits: (B,5)
      m_logits: (B,5,gh,gw)  where gh=H/patch, gw=W/patch (token grid)
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_labels: int = 5,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.gh = img_size // patch_size
        self.gw = img_size // patch_size
        self.num_patches = self.gh * self.gw
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # global head (CLS)
        self.head_global = nn.Linear(embed_dim, num_labels)

        # map head (tokens -> per-patch logits)
        self.head_map = nn.Linear(embed_dim, num_labels)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head_global.weight, std=0.02)
        nn.init.zeros_(self.head_global.bias)
        nn.init.trunc_normal_(self.head_map.weight, std=0.02)
        nn.init.zeros_(self.head_map.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        # patchify
        t = self.patch_embed(x)                # (B, D, gh, gw)
        t = t.flatten(2).transpose(1, 2)       # (B, N, D) N=gh*gw

        cls = self.cls_token.expand(B, -1, -1) # (B,1,D)
        z = torch.cat([cls, t], dim=1)         # (B,1+N,D)
        z = z + self.pos_embed[:, : z.size(1)]
        z = self.pos_drop(z)

        z = self.encoder(z)                    # (B,1+N,D)
        z = self.norm(z)

        # global
        cls_tok = z[:, 0]                      # (B,D)
        s_logits = self.head_global(cls_tok)   # (B,5)

        # map (tokens only)
        tok = z[:, 1:]                         # (B,N,D)
        m_logits_flat = self.head_map(tok)     # (B,N,5)
        m_logits = m_logits_flat.transpose(1, 2).contiguous().view(B, 5, self.gh, self.gw)  # (B,5,gh,gw)

        return s_logits, m_logits


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def multilabel_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    correct = (preds == targets).float()
    acc_per = correct.mean(dim=0)
    acc_macro = acc_per.mean().item()
    exact = (correct.min(dim=1).values).mean().item()
    out = {"acc_macro": float(acc_macro), "exact_match": float(exact)}
    for i, name in enumerate(LABEL_NAMES):
        out[f"acc_{name}"] = float(acc_per[i].item())
    return out


# ============================================================
# CKPT load (Phase-2a init)
# ============================================================
def load_phase2a_init(model: nn.Module, init_ckpt: str) -> Dict[str, Any]:
    """
    Loads a Phase-2a ckpt that was saved with keys:
      - state_dict (best/last)
    We load with strict=False because Phase-2b has an extra head_map.
    """
    ckpt = torch.load(init_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return {
        "init_ckpt": init_ckpt,
        "missing": missing[:20],
        "unexpected": unexpected[:20],
        "missing_n": len(missing),
        "unexpected_n": len(unexpected),
    }


# ============================================================
# Config
# ============================================================
@dataclass
class TrainCfg:
    data_root: str
    out_dir: str
    init_ckpt: str

    split_train: str
    split_val: str

    patch: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float

    use_amp: bool
    channels_last: bool
    tf32: bool

    vit_patch: int
    vit_dim: int
    vit_depth: int
    vit_heads: int

    max_per_action: int
    seed: int

    thr: float
    lambda_map: float
    map_warmup_epochs: int
    use_csd_mask: bool


def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", default="E:/ReAct-IR/data")
    ap.add_argument("--out_dir", default="E:/ReAct-IR/checkpoints/diagnoser")

    ap.add_argument("--init_ckpt", required=True, help="Phase-2a diagnoser_best.pth path")

    ap.add_argument("--split_train", default="train")
    ap.add_argument("--split_val", default="val")

    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=1)

    ap.add_argument("--vit_patch", type=int, default=16)
    ap.add_argument("--vit_dim", type=int, default=384)
    ap.add_argument("--vit_depth", type=int, default=6)
    ap.add_argument("--vit_heads", type=int, default=6)

    ap.add_argument("--max_per_action", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--lambda_map", type=float, default=1.0)
    ap.add_argument("--map_warmup_epochs", type=int, default=1, help="epochs to train cls only before adding map loss")
    ap.add_argument("--use_csd_mask", type=int, default=1)

    a = ap.parse_args()
    return TrainCfg(
        data_root=a.data_root,
        out_dir=a.out_dir,
        init_ckpt=a.init_ckpt,
        split_train=a.split_train,
        split_val=a.split_val,
        patch=int(a.patch),
        batch_size=int(a.batch_size),
        num_workers=int(a.num_workers),
        epochs=int(a.epochs),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        use_amp=bool(int(a.use_amp)),
        channels_last=bool(int(a.channels_last)),
        tf32=bool(int(a.tf32)),
        vit_patch=int(a.vit_patch),
        vit_dim=int(a.vit_dim),
        vit_depth=int(a.vit_depth),
        vit_heads=int(a.vit_heads),
        max_per_action=int(a.max_per_action),
        seed=int(a.seed),
        thr=float(a.thr),
        lambda_map=float(a.lambda_map),
        map_warmup_epochs=int(a.map_warmup_epochs),
        use_csd_mask=bool(int(a.use_csd_mask)),
    )


# ============================================================
# Train/Eval
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    thr: float,
    lambda_map: float,
    use_map_loss: bool,
) -> Dict[str, float]:
    model.train()
    bce = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    loss_cls_sum = 0.0
    loss_map_sum = 0.0
    n = 0

    t0 = time.time()
    pbar = tqdm(loader, ncols=140, desc="Train")

    for it, (inp, gt, y, masks, metas) in enumerate(pbar, start=1):
        inp = inp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            s_logits, m_logits = model(inp)

            loss_cls = bce(s_logits, y)

            loss_map = torch.zeros((), device=device)
            if use_map_loss:
                # only supervise snow map when mask exists
                # m_logits: (B,5,gh,gw) -> upsample to (B,5,H,W)
                m_up = F.interpolate(m_logits, size=(inp.shape[2], inp.shape[3]), mode="bilinear", align_corners=False)

                # build mask batch tensor for snow channel only
                # masks list items: None or (1,H,W)
                snow_masks = []
                snow_idx = []
                for bi, mk in enumerate(masks):
                    if mk is not None:
                        snow_masks.append(mk)   # (1,H,W)
                        snow_idx.append(bi)
                if len(snow_masks) > 0:
                    snow_masks = torch.stack(snow_masks, dim=0).to(device, non_blocking=True)  # (B2,1,H,W)
                    # pick corresponding predictions
                    pred_snow = m_up[snow_idx, SNOW_CH:SNOW_CH+1, :, :]  # (B2,1,H,W)
                    # BCE on map logits
                    loss_map = F.binary_cross_entropy_with_logits(pred_snow, snow_masks)

            loss = loss_cls + (lambda_map * loss_map)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        bs = int(inp.size(0))
        loss_sum += float(loss.item()) * bs
        loss_cls_sum += float(loss_cls.item()) * bs
        loss_map_sum += float(loss_map.item()) * bs
        n += bs

        with torch.no_grad():
            m = multilabel_metrics_from_logits(s_logits.detach(), y, thr=thr)

        elapsed = time.time() - t0
        it_per_sec = it / max(elapsed, 1e-6)
        eta_sec = (len(loader) - it) / max(it_per_sec, 1e-6)

        pbar.set_postfix({
            "L": f"{loss_sum/max(n,1):.4f}",
            "Lcls": f"{loss_cls_sum/max(n,1):.4f}",
            "Lmap": f"{loss_map_sum/max(n,1):.4f}" if use_map_loss else "0.0000",
            "acc": f"{m['acc_macro']:.3f}",
            "ETA": format_time(eta_sec),
        })

    out = {
        "loss": float(loss_sum / max(n, 1)),
        "loss_cls": float(loss_cls_sum / max(n, 1)),
        "loss_map": float(loss_map_sum / max(n, 1)),
    }
    out.update({k: float(v) for k, v in m.items()})
    return out


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    thr: float,
    lambda_map: float,
    use_map_loss: bool,
) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    loss_cls_sum = 0.0
    loss_map_sum = 0.0
    n = 0

    m_accum = None

    pbar = tqdm(loader, ncols=140, desc="Val ")
    for inp, gt, y, masks, metas in pbar:
        inp = inp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            s_logits, m_logits = model(inp)
            loss_cls = bce(s_logits, y)

            loss_map = torch.zeros((), device=device)
            if use_map_loss:
                m_up = F.interpolate(m_logits, size=(inp.shape[2], inp.shape[3]), mode="bilinear", align_corners=False)
                snow_masks = []
                snow_idx = []
                for bi, mk in enumerate(masks):
                    if mk is not None:
                        snow_masks.append(mk)
                        snow_idx.append(bi)
                if len(snow_masks) > 0:
                    snow_masks = torch.stack(snow_masks, dim=0).to(device, non_blocking=True)
                    pred_snow = m_up[snow_idx, SNOW_CH:SNOW_CH+1, :, :]
                    loss_map = F.binary_cross_entropy_with_logits(pred_snow, snow_masks)

            loss = loss_cls + (lambda_map * loss_map)

        bs = int(inp.size(0))
        loss_sum += float(loss.item()) * bs
        loss_cls_sum += float(loss_cls.item()) * bs
        loss_map_sum += float(loss_map.item()) * bs
        n += bs

        m = multilabel_metrics_from_logits(s_logits, y, thr=thr)
        m_accum = m if m_accum is None else {k: (m_accum[k] + m[k]) for k in m_accum.keys()}

        pbar.set_postfix({
            "L": f"{loss_sum/max(n,1):.4f}",
            "Lcls": f"{loss_cls_sum/max(n,1):.4f}",
            "Lmap": f"{loss_map_sum/max(n,1):.4f}" if use_map_loss else "0.0000",
            "acc": f"{m['acc_macro']:.3f}",
        })

    out = {
        "loss": float(loss_sum / max(n, 1)),
        "loss_cls": float(loss_cls_sum / max(n, 1)),
        "loss_map": float(loss_map_sum / max(n, 1)),
    }
    if m_accum is not None:
        out.update({k: float(v / max(len(loader), 1)) for k, v in m_accum.items()})
    return out


def main():
    cfg = parse_args()
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    torch.backends.cudnn.benchmark = True
    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"[Speed] tf32={cfg.tf32} channels_last={cfg.channels_last} amp={cfg.use_amp}")

    safe_makedirs(cfg.out_dir)

    # ---------------- datasets ----------------
    ds_tr = DiagnoserMapDataset(
        data_root=cfg.data_root,
        split=cfg.split_train,
        patch=cfg.patch,
        augment=True,
        max_per_action=cfg.max_per_action,
        seed=cfg.seed,
        strict_size_check=False,
        use_csd_mask=cfg.use_csd_mask,
    )
    ds_va = DiagnoserMapDataset(
        data_root=cfg.data_root,
        split=cfg.split_val,
        patch=cfg.patch,
        augment=False,
        max_per_action=cfg.max_per_action if cfg.max_per_action > 0 else -1,
        seed=cfg.seed + 1,
        strict_size_check=False,
        use_csd_mask=cfg.use_csd_mask,
    )
    print("[Train] total:", len(ds_tr), "by_action:", ds_tr.stats_by_action)
    print("[Val  ] total:", len(ds_va), "by_action:", ds_va.stats_by_action)

    loader_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_diagnoser_map,
        persistent_workers=(cfg.num_workers > 0),
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_diagnoser_map,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---------------- model ----------------
    model = TinyViTDiagnoserWithMap(
        img_size=cfg.patch,
        patch_size=cfg.vit_patch,
        embed_dim=cfg.vit_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_heads,
        num_labels=len(LABEL_NAMES),
    )
    init_info = load_phase2a_init(model, cfg.init_ckpt)
    print("[Init] loaded with strict=False")
    print(f"  missing={init_info['missing_n']} unexpected={init_info['unexpected_n']}")
    if init_info["missing_n"] > 0:
        print("  example missing:", init_info["missing"][:10])
    if init_info["unexpected_n"] > 0:
        print("  example unexpected:", init_info["unexpected"][:10])

    model.to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val = 1e9
    best_path = os.path.join(cfg.out_dir, "diagnoser_map_best.pth")
    last_path = os.path.join(cfg.out_dir, "diagnoser_map_last.pth")

    print("\n[Train] start")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        use_map_loss = (epoch > cfg.map_warmup_epochs)

        tr = train_one_epoch(
            model, loader_tr, opt, scaler, device,
            use_amp=cfg.use_amp, thr=cfg.thr,
            lambda_map=cfg.lambda_map, use_map_loss=use_map_loss
        )
        va = eval_one_epoch(
            model, loader_va, device,
            use_amp=cfg.use_amp, thr=cfg.thr,
            lambda_map=cfg.lambda_map, use_map_loss=use_map_loss
        )

        print(
            f"\n[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train: L={tr['loss']:.4f} (cls={tr['loss_cls']:.4f} map={tr['loss_map']:.4f}) acc={tr['acc_macro']:.3f} | "
            f"val: L={va['loss']:.4f} (cls={va['loss_cls']:.4f} map={va['loss_map']:.4f}) acc={va.get('acc_macro',0):.3f} | "
            f"map_on={int(use_map_loss)} time={format_time(time.time()-t0)}"
        )

        ckpt = {
            "epoch": epoch,
            "cfg": vars(cfg),
            "label_names": LABEL_NAMES,
            "init_ckpt": cfg.init_ckpt,
            "state_dict": model.state_dict(),
            "train": tr,
            "val": va,
        }
        torch.save(ckpt, last_path)

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ckpt, best_path)
            print(f"[Save] BEST -> {best_path} (val_loss={best_val:.6f})")

    print("\n[Train] finished")
    print("[Output] best =", best_path)
    print("[Output] last =", last_path)


if __name__ == "__main__":
    main()

"""
python scripts/train_diagnoser_map.py ^
  --data_root "E:/ReAct-IR/data" ^
  --out_dir "E:/ReAct-IR/checkpoints/diagnoser" ^
  --init_ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" ^
  --split_train train --split_val val ^
  --epochs 10 --batch_size 16 --num_workers 4 ^
  --lambda_map 1.0 --map_warmup_epochs 1 ^
  --use_csd_mask 1

"""