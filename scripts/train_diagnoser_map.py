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

from datasets.lora_dataset import ActionPairConfig, build_action_pairs

# ============================================================
# Label / Action mapping
# ============================================================
ACTIONS_INTERNAL = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
ACTION_TO_LABEL_INDEX = {
    "A_DEBLUR": 0,
    "A_DERAIN": 1,
    "A_DESNOW": 2,
    "A_DEHAZE": 3,
    "A_DEDROP": 4,
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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


@torch.no_grad()
def multilabel_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    """
    logits: (B,5)
    targets: (B,5) in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    correct = (preds == targets).float()  # (B,5)
    acc_per_label = correct.mean(dim=0)   # (5,)
    acc_macro = acc_per_label.mean().item()
    exact = (correct.min(dim=1).values).mean().item()

    out = {"acc_macro": float(acc_macro), "exact_match": float(exact)}
    for i, name in enumerate(LABEL_NAMES):
        out[f"acc_{name}"] = float(acc_per_label[i].item())
    return out


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return x


def pil_to_tensor_mask(img: Image.Image) -> torch.Tensor:
    # grayscale -> (1,H,W), normalized to [0,1]
    arr = np.array(img).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).contiguous()
    return x


def _random_crop_params(H: int, W: int, patch: int, rng: random.Random):
    if H == patch and W == patch:
        return 0, 0
    y0 = rng.randint(0, max(0, H - patch))
    x0 = rng.randint(0, max(0, W - patch))
    return y0, x0


def _center_crop_params(H: int, W: int, patch: int):
    y0 = max(0, (H - patch) // 2)
    x0 = max(0, (W - patch) // 2)
    return y0, x0


def apply_crop_flip_rot(
    img: Image.Image,
    patch: int,
    augment: bool,
    rng: random.Random,
    crop_mode: str = "random",
    same_transform: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Apply crop + (optional) flip/rot. If same_transform is provided, reuse it.
    Returns transformed image and the used transform dict.
    """
    img = img.convert("RGB") if img.mode != "RGB" else img
    W, H = img.size

    # resize up if too small
    if H < patch or W < patch:
        img = img.resize((patch, patch), resample=Image.BILINEAR)
        W, H = img.size

    if same_transform is None:
        if crop_mode == "center" or (not augment):
            y0, x0 = _center_crop_params(H, W, patch)
        else:
            y0, x0 = _random_crop_params(H, W, patch, rng)

        do_hflip = augment and (rng.random() < 0.5)
        do_vflip = augment and (rng.random() < 0.2)
        rot_k = rng.randint(0, 3) if augment else 0  # 0,1,2,3 * 90deg

        tfm = {"y0": y0, "x0": x0, "do_hflip": do_hflip, "do_vflip": do_vflip, "rot_k": rot_k}
    else:
        tfm = same_transform
        y0, x0 = int(tfm["y0"]), int(tfm["x0"])
        do_hflip = bool(tfm["do_hflip"])
        do_vflip = bool(tfm["do_vflip"])
        rot_k = int(tfm["rot_k"])

    img = img.crop((x0, y0, x0 + patch, y0 + patch))

    if do_hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if do_vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if rot_k > 0:
        img = img.rotate(90 * rot_k, expand=False)

    return img, tfm


def apply_crop_flip_rot_mask(
    img: Image.Image,
    patch: int,
    tfm: Dict[str, Any],
) -> Image.Image:
    """
    Apply SAME transform used for RGB image to mask image.
    """
    W, H = img.size

    if H < patch or W < patch:
        img = img.resize((patch, patch), resample=Image.NEAREST)
        W, H = img.size

    y0, x0 = int(tfm["y0"]), int(tfm["x0"])
    do_hflip = bool(tfm["do_hflip"])
    do_vflip = bool(tfm["do_vflip"])
    rot_k = int(tfm["rot_k"])

    img = img.crop((x0, y0, x0 + patch, y0 + patch))

    if do_hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if do_vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if rot_k > 0:
        img = img.rotate(90 * rot_k, expand=False)

    return img


def infer_csd_mask_path(inp_path: str) -> Optional[str]:
    """
    CSD structure:
      .../CSD/Train/Snow/xxx.tif
      .../CSD/Train/Mask/xxx.tif
      .../CSD/Test/Snow/xxx.tif
      .../CSD/Test/Mask/xxx.tif
    We replace "\Snow\" with "\Mask\" (case-insensitive).
    """
    p = inp_path.replace("\\", "/")
    lower = p.lower()
    if "/csd/" not in lower:
        return None
    # find "/snow/" segment
    if "/snow/" not in lower:
        return None

    # replace only the last occurrence robustly
    parts = p.split("/")
    # find 'Snow' component index
    idx = None
    for i in range(len(parts)):
        if parts[i].lower() == "snow":
            idx = i
    if idx is None:
        return None
    parts[idx] = "Mask"
    mask_path = "/".join(parts)

    # try same extension
    if os.path.exists(mask_path):
        return mask_path

    # fallback: try png/tif variants if extension differs
    base, ext = os.path.splitext(mask_path)
    for e in [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]:
        cand = base + e
        if os.path.exists(cand):
            return cand
    return None


# ============================================================
# Split helper (supports val from train)
# ============================================================
def split_train_val_per_action(
    pairs: List[Tuple[str, str, Dict[str, Any]]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, Dict[str, Any]]]]:
    """
    Deterministic split per action (so each class has val samples).
    """
    rng = random.Random(seed)
    by_action: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {a: [] for a in ACTIONS_INTERNAL}
    for it in pairs:
        meta = it[2]
        a = meta.get("action_internal", None)
        if a in by_action:
            by_action[a].append(it)

    tr_all, va_all = [], []
    for a, items in by_action.items():
        rng.shuffle(items)
        n = len(items)
        nv = int(round(n * val_ratio))
        nv = max(1, nv) if n > 1 else 0
        va = items[:nv]
        tr = items[nv:]
        tr_all.extend(tr)
        va_all.extend(va)

    rng.shuffle(tr_all)
    rng.shuffle(va_all)
    return tr_all, va_all


# ============================================================
# Dataset (NO LoRAPairedDataset usage)
# ============================================================
class DiagnoserMapDataset(Dataset):
    """
    Returns:
      inp: (3,H,W) float
      gt : (3,H,W) float  (not strictly required, but kept)
      y  : (5,) one-hot global label
      mask_map: (5,H,W) float target map (only snow uses CSD mask; else zeros)
      mask_valid: (1,) 1 if map supervision is valid else 0
      meta: dict
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        patch: int,
        augment: bool,
        max_per_action: int,
        seed: int,
        use_csd_mask: bool,
        val_from_train: bool = False,
        val_ratio: float = 0.2,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.patch = int(patch)
        self.augment = bool(augment)
        self.max_per_action = int(max_per_action)
        self.seed = int(seed)
        self.use_csd_mask = bool(use_csd_mask)

        rng = random.Random(self.seed)

        # collect per action from build_action_pairs(train|test only)
        # If split == "val", we will build from train then split.
        build_split = split
        if split == "val":
            build_split = "train"

        all_pairs: List[Tuple[str, str, Dict[str, Any]]] = []
        self.stats_by_action: Dict[str, int] = {}

        for action in ACTIONS_INTERNAL:
            pair_cfg = ActionPairConfig(action=action, split=build_split, data_root=data_root)
            pairs, _ = build_action_pairs(pair_cfg)

            # attach action into meta (ensure dict)
            fixed = []
            for inp_path, gt_path, meta in pairs:
                if not isinstance(meta, dict):
                    # hard guard (shouldn't happen with build_action_pairs)
                    meta = {"dataset": "UNKNOWN"}
                m2 = dict(meta)
                m2["action_internal"] = action
                fixed.append((inp_path, gt_path, m2))

            # cap per action (balance)
            if self.max_per_action > 0 and len(fixed) > self.max_per_action:
                rng.shuffle(fixed)
                fixed = fixed[: self.max_per_action]

            all_pairs.extend(fixed)
            self.stats_by_action[action] = len(fixed)

        rng.shuffle(all_pairs)

        # If val split requested, do deterministic split per action
        if split == "val":
            tr, va = split_train_val_per_action(all_pairs, val_ratio=val_ratio, seed=self.seed)
            self.pairs = va
        elif val_from_train and split == "train":
            tr, va = split_train_val_per_action(all_pairs, val_ratio=val_ratio, seed=self.seed)
            self.pairs = tr
        else:
            self.pairs = all_pairs

        self.rng = random.Random(self.seed + (0 if split == "train" else 999))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        inp_path, gt_path, meta = self.pairs[idx]
        assert isinstance(meta, dict), "meta must be dict"

        # load images
        inp_img = Image.open(inp_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        # crop/augment with SAME transform for inp and gt
        crop_mode = "random" if self.augment else "center"
        inp_img, tfm = apply_crop_flip_rot(inp_img, self.patch, self.augment, self.rng, crop_mode=crop_mode)
        gt_img, _ = apply_crop_flip_rot(gt_img, self.patch, self.augment, self.rng, crop_mode=crop_mode, same_transform=tfm)

        inp = pil_to_tensor_rgb(inp_img)  # (3,H,W)
        gt = pil_to_tensor_rgb(gt_img)

        # global one-hot label
        action = meta.get("action_internal")
        y = torch.zeros(5, dtype=torch.float32)
        y[ACTION_TO_LABEL_INDEX[action]] = 1.0

        # map target
        map_t = torch.zeros(5, self.patch, self.patch, dtype=torch.float32)
        mask_valid = torch.zeros(1, dtype=torch.float32)

        if self.use_csd_mask and action == "A_DESNOW":
            # try infer CSD mask path from inp_path
            mp = infer_csd_mask_path(inp_path)
            if mp is not None and os.path.exists(mp):
                m_img = Image.open(mp).convert("L")
                m_img = apply_crop_flip_rot_mask(m_img, self.patch, tfm)
                m = pil_to_tensor_mask(m_img).clamp(0, 1)  # (1,H,W)
                map_t[2:3] = m  # snow channel index = 2
                mask_valid[0] = 1.0
            else:
                # CSD mask missing => no map supervision
                mask_valid[0] = 0.0

        return inp, gt, y, map_t, mask_valid, meta


def collate_diagnoser_map(batch):
    inp = torch.stack([b[0] for b in batch], dim=0)          # (B,3,H,W)
    gt = torch.stack([b[1] for b in batch], dim=0)           # (B,3,H,W)
    y = torch.stack([b[2] for b in batch], dim=0)            # (B,5)
    maps = torch.stack([b[3] for b in batch], dim=0)         # (B,5,H,W)
    valid = torch.stack([b[4] for b in batch], dim=0)         # (B,1)
    metas = [b[5] for b in batch]
    return inp, gt, y, maps, valid, metas


# ============================================================
# Model: TinyViT with Global head + Map head
# ============================================================
class TinyViTDiagnoserMap(nn.Module):
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
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

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

        # global head
        self.head_global = nn.Linear(embed_dim, num_labels)

        # map head: token grid -> 5ch map at patch resolution
        self.head_map = nn.Conv2d(embed_dim, num_labels, kernel_size=1, stride=1, padding=0, bias=True)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head_global.weight, std=0.02)
        nn.init.zeros_(self.head_global.bias)
        nn.init.trunc_normal_(self.head_map.weight, std=0.02)
        nn.init.zeros_(self.head_map.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          logits_global: (B,5)
          logits_map: (B,5,H,W)  (upsampled to img_size)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "input must be cropped to img_size"

        # patch embed
        t = self.patch_embed(x)                           # (B,D,g,g)
        t_flat = t.flatten(2).transpose(1, 2)            # (B,N,D)

        cls = self.cls_token.expand(B, -1, -1)           # (B,1,D)
        z = torch.cat([cls, t_flat], dim=1)              # (B,1+N,D)
        z = z + self.pos_embed[:, : z.size(1)]
        z = self.pos_drop(z)

        z = self.encoder(z)                              # (B,1+N,D)

        # global
        g = self.norm(z[:, 0])                           # (B,D)
        logits_global = self.head_global(g)              # (B,5)

        # map tokens (exclude cls)
        tokens = z[:, 1:]                                # (B,N,D)
        tokens = tokens.transpose(1, 2).contiguous()     # (B,D,N)
        tokens = tokens.view(B, -1, self.grid, self.grid)  # (B,D,g,g)

        logits_low = self.head_map(tokens)               # (B,5,g,g)
        logits_map = F.interpolate(logits_low, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return logits_global, logits_map


def load_init_ckpt(model: nn.Module, ckpt_path: str):
    """
    Supports:
      - train_diagnoser.py ckpt with keys: head.weight/head.bias
      - train_diagnoser_map.py ckpt with keys: head_global.*, head_map.*
    We remap head.* -> head_global.* when needed.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # remap old head -> head_global
    if "head.weight" in sd and "head_global.weight" not in sd:
        sd2 = dict(sd)
        sd2["head_global.weight"] = sd2.pop("head.weight")
        sd2["head_global.bias"] = sd2.pop("head.bias")
        sd = sd2

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


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

    val_from_train: bool
    val_ratio: float


def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", default="E:/ReAct-IR/data")
    ap.add_argument("--out_dir", default="E:/ReAct-IR/checkpoints/diagnoser")
    ap.add_argument("--init_ckpt", default="E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth")

    ap.add_argument("--split_train", default="train")
    ap.add_argument("--split_val", default="test", help="train|test|val (val uses train split internally)")

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

    ap.add_argument("--max_per_action", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thr", type=float, default=0.5)

    ap.add_argument("--lambda_map", type=float, default=1.0)
    ap.add_argument("--map_warmup_epochs", type=int, default=1)
    ap.add_argument("--use_csd_mask", type=int, default=1)

    ap.add_argument("--val_from_train", type=int, default=0, help="if 1, train split also removes val_ratio from train per-action")
    ap.add_argument("--val_ratio", type=float, default=0.2)

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
        val_from_train=bool(int(a.val_from_train)),
        val_ratio=float(a.val_ratio),
    )


# ============================================================
# Train / Eval
# ============================================================
def map_loss_weight(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    thr: float,
    lambda_map: float,
    map_w: float,
) -> Dict[str, float]:
    model.train()
    bce = nn.BCEWithLogitsLoss()
    bce_map = nn.BCEWithLogitsLoss(reduction="none")

    loss_sum = 0.0
    loss_g_sum = 0.0
    loss_m_sum = 0.0
    n = 0

    t0 = time.time()
    pbar = tqdm(loader, ncols=140, desc="Train")

    last_metrics = {}

    for it, (inp, gt, y, maps, valid, metas) in enumerate(pbar, start=1):
        inp = inp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        maps = maps.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)  # (B,1)

        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            logits_g, logits_m = model(inp)  # (B,5), (B,5,H,W)

            loss_g = bce(logits_g, y)

            # map loss only where valid==1 (CSD snow samples)
            # compute per-pixel BCE and average
            per = bce_map(logits_m, maps)  # (B,5,H,W)
            # apply sample validity
            v = valid.view(-1, 1, 1, 1)    # (B,1,1,1)
            per = per * v
            denom = v.sum() * per.shape[1] * per.shape[2] * per.shape[3]
            loss_m = per.sum() / (denom + 1e-6)

            loss = loss_g + (lambda_map * map_w) * loss_m

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        B = inp.size(0)
        loss_sum += float(loss.item()) * B
        loss_g_sum += float(loss_g.item()) * B
        loss_m_sum += float(loss_m.item()) * B
        n += B

        with torch.no_grad():
            m = multilabel_metrics_from_logits(logits_g.detach(), y, thr=thr)
            last_metrics = m

        elapsed = time.time() - t0
        it_per_sec = it / max(elapsed, 1e-6)
        eta_sec = (len(loader) - it) / max(it_per_sec, 1e-6)

        pbar.set_postfix({
            "L": f"{loss_sum/max(n,1):.4f}",
            "Lg": f"{loss_g_sum/max(n,1):.4f}",
            "Lm": f"{loss_m_sum/max(n,1):.4f}",
            "mw": f"{map_w:.2f}",
            "acc": f"{m['acc_macro']:.3f}",
            "ex": f"{m['exact_match']:.3f}",
            "ETA": format_time(eta_sec),
        })

    out = {
        "loss": float(loss_sum / max(n, 1)),
        "loss_global": float(loss_g_sum / max(n, 1)),
        "loss_map": float(loss_m_sum / max(n, 1)),
    }
    out.update({k: float(v) for k, v in last_metrics.items()})
    return out


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    thr: float,
    lambda_map: float,
    map_w: float,
) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    bce_map = nn.BCEWithLogitsLoss(reduction="none")

    loss_sum = 0.0
    loss_g_sum = 0.0
    loss_m_sum = 0.0
    n = 0

    m_accum = None  # type: Optional[Dict[str, float]]

    pbar = tqdm(loader, ncols=140, desc="Val ")
    for inp, gt, y, maps, valid, metas in pbar:
        inp = inp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        maps = maps.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            logits_g, logits_m = model(inp)

            loss_g = bce(logits_g, y)

            per = bce_map(logits_m, maps)
            v = valid.view(-1, 1, 1, 1)
            per = per * v
            denom = v.sum() * per.shape[1] * per.shape[2] * per.shape[3]
            loss_m = per.sum() / (denom + 1e-6)

            loss = loss_g + (lambda_map * map_w) * loss_m

        B = inp.size(0)
        loss_sum += float(loss.item()) * B
        loss_g_sum += float(loss_g.item()) * B
        loss_m_sum += float(loss_m.item()) * B
        n += B

        m = multilabel_metrics_from_logits(logits_g, y, thr=thr)
        if m_accum is None:
            m_accum = {k: float(v) for k, v in m.items()}
        else:
            for k in m_accum.keys():
                m_accum[k] += float(m[k])

        pbar.set_postfix({
            "L": f"{loss_sum/max(n,1):.4f}",
            "Lg": f"{loss_g_sum/max(n,1):.4f}",
            "Lm": f"{loss_m_sum/max(n,1):.4f}",
            "acc": f"{m['acc_macro']:.3f}",
            "ex": f"{m['exact_match']:.3f}",
        })

    out = {
        "loss": float(loss_sum / max(n, 1)),
        "loss_global": float(loss_g_sum / max(n, 1)),
        "loss_map": float(loss_m_sum / max(n, 1)),
    }
    if m_accum is not None:
        for k in m_accum.keys():
            out[k] = float(m_accum[k] / max(len(loader), 1))
    return out


# ============================================================
# Main
# ============================================================
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

    # ---- dataset
    # split_val supports val (internal split from train)
    ds_tr = DiagnoserMapDataset(
        data_root=cfg.data_root,
        split=cfg.split_train,
        patch=cfg.patch,
        augment=True,
        max_per_action=cfg.max_per_action,
        seed=cfg.seed,
        use_csd_mask=cfg.use_csd_mask,
        val_from_train=cfg.val_from_train,
        val_ratio=cfg.val_ratio,
    )

    if cfg.split_val == "val":
        ds_va = DiagnoserMapDataset(
            data_root=cfg.data_root,
            split="val",
            patch=cfg.patch,
            augment=False,
            max_per_action=cfg.max_per_action,
            seed=cfg.seed + 1,
            use_csd_mask=cfg.use_csd_mask,
            val_from_train=False,
            val_ratio=cfg.val_ratio,
        )
    else:
        ds_va = DiagnoserMapDataset(
            data_root=cfg.data_root,
            split=cfg.split_val,
            patch=cfg.patch,
            augment=False,
            max_per_action=cfg.max_per_action,
            seed=cfg.seed + 1,
            use_csd_mask=cfg.use_csd_mask,
            val_from_train=False,
            val_ratio=cfg.val_ratio,
        )

    print("[Train] total:", len(ds_tr), "by_action:", ds_tr.stats_by_action)
    print("[Val  ] total:", len(ds_va), "by_action:", ds_va.stats_by_action)
    print("[Map] use_csd_mask =", cfg.use_csd_mask)

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

    # ---- model
    model = TinyViTDiagnoserMap(
        img_size=cfg.patch,
        patch_size=cfg.vit_patch,
        embed_dim=cfg.vit_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_heads,
        num_labels=len(LABEL_NAMES),
    )

    model.to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # ---- init ckpt
    if cfg.init_ckpt and os.path.exists(cfg.init_ckpt):
        missing, unexpected = load_init_ckpt(model, cfg.init_ckpt)
        print("[Init] loaded with strict=False")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  example missing:", missing[:10])
        if len(unexpected) > 0:
            print("  example unexpected:", unexpected[:10])
    else:
        print("[Init] skipped (ckpt not found)")

    # ---- opt
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val = 1e9
    best_path = os.path.join(cfg.out_dir, "diagnoser_map_best.pth")
    last_path = os.path.join(cfg.out_dir, "diagnoser_map_last.pth")

    print("\n[Train] start")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        mw = map_loss_weight(epoch, cfg.map_warmup_epochs)

        tr = train_one_epoch(
            model, loader_tr, opt, scaler, device,
            use_amp=cfg.use_amp, thr=cfg.thr,
            lambda_map=cfg.lambda_map, map_w=mw
        )
        va = eval_one_epoch(
            model, loader_va, device,
            use_amp=cfg.use_amp, thr=cfg.thr,
            lambda_map=cfg.lambda_map, map_w=mw
        )

        print(f"\n[Epoch {epoch:03d}/{cfg.epochs}] "
              f"train: L={tr['loss']:.4f} (Lg={tr['loss_global']:.4f}, Lm={tr['loss_map']:.4f}) acc={tr['acc_macro']:.3f} ex={tr['exact_match']:.3f} | "
              f"val: L={va['loss']:.4f} (Lg={va['loss_global']:.4f}, Lm={va['loss_map']:.4f}) acc={va.get('acc_macro',0):.3f} ex={va.get('exact_match',0):.3f} | "
              f"mw={mw:.2f} time={format_time(time.time()-t0)}")

        ckpt = {
            "epoch": epoch,
            "cfg": vars(cfg),
            "label_names": LABEL_NAMES,
            "actions": ACTIONS_INTERNAL,
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
