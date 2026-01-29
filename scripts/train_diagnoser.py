# 라벨 JSON으로 Diagnoser 학습
# E:/ReAct-IR/scripts/train_diagnoser.py
# 상태인식 -> VLM (CLIP, BLIP, vision encoder) | 순수 ViT or ViT-H/14 같은 구조
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

from datasets.lora_dataset import (
    ActionPairConfig,
    build_action_pairs,
)


# --------------------------------------------------
# Optional skimage metrics (not required)
# --------------------------------------------------
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False


# ============================================================
# Action / Label mapping
# ============================================================
# Tool action names you used in ToolBank (internal)
ACTIONS_INTERNAL = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]

# label order: [blur, rain, snow, haze, drop]
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
ACTION_TO_LABEL_INDEX = {
    "A_DEBLUR": 0,
    "A_DERAIN": 1,
    "A_DESNOW": 2,
    "A_DEHAZE": 3,
    "A_DEDROP": 4,
}

# folder aliases are what you used for LoRA ckpt dirs
ALIAS_TO_INTERNAL = {
    "deblur": "A_DEBLUR",
    "derain": "A_DERAIN",
    "desnow": "A_DESNOW",
    "dehaze": "A_DEHAZE",
    "dedrop": "A_DEDROP",
}


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

    # per-label accuracy
    correct = (preds == targets).float()  # (B,5)
    acc_per_label = correct.mean(dim=0)   # (5,)
    acc_macro = acc_per_label.mean().item()

    # exact match ratio (all 5 labels correct)
    exact = (correct.min(dim=1).values).mean().item()

    out = {"acc_macro": float(acc_macro), "exact_match": float(exact)}
    for i, name in enumerate(LABEL_NAMES):
        out[f"acc_{name}"] = float(acc_per_label[i].item())
    return out


# ============================================================
# Dataset: reuse LoRAPairedDataset but only return input + label
# ============================================================
class DiagnoserDataset(Dataset):
    """
    build_action_pairs()로 얻은 inp_path를 직접 읽어서 Tensor로 반환한다.
    LoRAPairedDataset 반환 타입(문자열 토큰 등) 이슈를 원천 차단.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        patch: int,
        augment: bool,
        actions_internal: List[str],
        max_per_action: int = -1,
        seed: int = 123,
    ):
        super().__init__()
        self.patch = int(patch)
        self.augment = bool(augment)
        self.seed = int(seed)

        rng = random.Random(self.seed)

        self.items: List[Tuple[str, str]] = []  # (inp_path, action_internal)
        self.stats_by_action: Dict[str, int] = {}

        for action in actions_internal:
            pair_cfg = ActionPairConfig(action=action, split=split, data_root=data_root)
            pairs, _ = build_action_pairs(pair_cfg)

            if max_per_action > 0 and len(pairs) > max_per_action:
                pairs = pairs.copy()
                rng.shuffle(pairs)
                pairs = pairs[:max_per_action]

            for inp_path, gt_path, meta in pairs:
                # inp_path should be an actual path
                self.items.append((inp_path, action))

            self.stats_by_action[action] = len(pairs)

        rng.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def _load_rgb_tensor(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")

        W, H = img.size
        ph = pw = self.patch

        # crop or resize
        if (H >= ph) and (W >= pw):
            if self.augment:
                x0 = random.randint(0, W - pw)
                y0 = random.randint(0, H - ph)
            else:
                x0 = max(0, (W - pw) // 2)
                y0 = max(0, (H - ph) // 2)
            img = img.crop((x0, y0, x0 + pw, y0 + ph))
        else:
            img = img.resize((pw, ph), resample=Image.BILINEAR)

        # simple aug
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        arr = np.array(img).astype(np.float32) / 255.0  # HWC
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        return x

    def __getitem__(self, idx: int):
        inp_path, action = self.items[idx]
        inp = self._load_rgb_tensor(inp_path)

        y = torch.zeros(5, dtype=torch.float32)
        y[ACTION_TO_LABEL_INDEX[action]] = 1.0

        meta = {"inp_path": inp_path, "action_internal": action}
        return inp, y, meta




def collate_diagnoser(batch):
    inps = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return inps, ys, metas




# ============================================================
# Model: simple ViT-like (patch embed + transformer blocks) OR small CNN
# ============================================================
class TinyViTDiagnoser(nn.Module):
    """
    A lightweight ViT-like classifier for 256x256 patches.
    This avoids external deps (timm). Works well enough for score-vector diagnosis.
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
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # patch embedding via conv
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        # cls token + pos embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # transformer encoder blocks
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
        self.head = nn.Linear(embed_dim, num_labels)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        B, C, H, W = x.shape
        # patchify
        x = self.patch_embed(x)  # (B,embed_dim,H/p,W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,embed_dim)
        x = torch.cat([cls, x], dim=1)          # (B,1+num_patches,embed_dim)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)

        x = self.encoder(x)                     # (B,1+num_patches,embed_dim)
        x = self.norm(x[:, 0])                  # take cls token
        logits = self.head(x)                   # (B,5)
        return logits


class TinyCNNDiagnoser(nn.Module):
    """
    Simple CNN baseline (faster, often surprisingly strong).
    """
    def __init__(self, num_labels: int = 5, width: int = 64):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True),          # /2
            nn.Conv2d(w, 2*w, 3, 2, 1), nn.ReLU(inplace=True),        # /4
            nn.Conv2d(2*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /8
            nn.Conv2d(4*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /16
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(4*w, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x).flatten(1)
        return self.head(f)


# ============================================================
# Config
# ============================================================
@dataclass
class TrainCfg:
    data_root: str
    out_dir: str

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

    model: str  # "vit" or "cnn"
    vit_patch: int
    vit_dim: int
    vit_depth: int
    vit_heads: int
    cnn_width: int

    max_per_action: int
    seed: int
    thr: float


def parse_args() -> TrainCfg:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", default="E:/ReAct-IR/data")
    ap.add_argument("--out_dir", default="E:/ReAct-IR/checkpoints/diagnoser")

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

    ap.add_argument("--model", default="vit", choices=["vit", "cnn"])
    ap.add_argument("--vit_patch", type=int, default=16)
    ap.add_argument("--vit_dim", type=int, default=384)
    ap.add_argument("--vit_depth", type=int, default=6)
    ap.add_argument("--vit_heads", type=int, default=6)

    ap.add_argument("--cnn_width", type=int, default=64)

    ap.add_argument("--max_per_action", type=int, default=-1, help="optional cap per action for balancing; -1 = no cap")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thr", type=float, default=0.5)

    a = ap.parse_args()
    return TrainCfg(
        data_root=a.data_root,
        out_dir=a.out_dir,
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
        model=str(a.model),
        vit_patch=int(a.vit_patch),
        vit_dim=int(a.vit_dim),
        vit_depth=int(a.vit_depth),
        vit_heads=int(a.vit_heads),
        cnn_width=int(a.cnn_width),
        max_per_action=int(a.max_per_action),
        seed=int(a.seed),
        thr=float(a.thr),
    )


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    thr: float,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    n = 0

    # BCE with logits for multi-label
    bce = nn.BCEWithLogitsLoss()

    t0 = time.time()
    pbar = tqdm(loader, ncols=140, desc="Train")

    for it, (inp, y, metas) in enumerate(pbar, start=1):
        inp = inp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            logits = model(inp)
            loss = bce(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loss_sum += float(loss.item()) * inp.size(0)
        n += int(inp.size(0))

        # quick metrics
        with torch.no_grad():
            m = multilabel_metrics_from_logits(logits.detach(), y, thr=thr)

        elapsed = time.time() - t0
        it_per_sec = it / max(elapsed, 1e-6)
        eta_sec = (len(loader) - it) / max(it_per_sec, 1e-6)

        pbar.set_postfix({
            "L": f"{loss_sum/max(n,1):.4f}",
            "acc": f"{m['acc_macro']:.3f}",
            "ex": f"{m['exact_match']:.3f}",
            "ETA": format_time(eta_sec),
        })

    out = {"loss": float(loss_sum / max(n, 1))}
    out.update({k: float(v) for k, v in m.items()})
    return out


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    thr: float,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n = 0
    bce = nn.BCEWithLogitsLoss()

    m_accum = None  # type: Optional[Dict[str, float]]

    pbar = tqdm(loader, ncols=140, desc="Val ")
    for inp, y, metas in pbar:
        inp = inp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            logits = model(inp)
            loss = bce(logits, y)

        loss_sum += float(loss.item()) * inp.size(0)
        n += int(inp.size(0))

        m = multilabel_metrics_from_logits(logits, y, thr=thr)
        m_accum = m if m_accum is None else {k: (m_accum[k] + m[k]) for k in m_accum.keys()}

        pbar.set_postfix({
            "L": f"{loss_sum/max(n,1):.4f}",
            "acc": f"{m['acc_macro']:.3f}",
            "ex": f"{m['exact_match']:.3f}",
        })

    out = {"loss": float(loss_sum / max(n, 1))}
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
    print("[Labels]", LABEL_NAMES)
    print("[Actions]", ACTIONS_INTERNAL)

    safe_makedirs(cfg.out_dir)

    # ---------------- dataset ----------------
    ds_tr = DiagnoserDataset(
        data_root=cfg.data_root,
        split=cfg.split_train,
        patch=cfg.patch,
        augment=True,
        actions_internal=ACTIONS_INTERNAL,
        max_per_action=cfg.max_per_action,
        seed=cfg.seed,
    )
    ds_va = DiagnoserDataset(
        data_root=cfg.data_root,
        split=cfg.split_val,
        patch=cfg.patch,
        augment=False,
        actions_internal=ACTIONS_INTERNAL,
        max_per_action=cfg.max_per_action if cfg.max_per_action > 0 else -1,
        seed=cfg.seed + 1,
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
        collate_fn=collate_diagnoser,
        persistent_workers=(cfg.num_workers > 0),
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_diagnoser,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---------------- model ----------------
    if cfg.model == "vit":
        model = TinyViTDiagnoser(
            img_size=cfg.patch,
            patch_size=cfg.vit_patch,
            embed_dim=cfg.vit_dim,
            depth=cfg.vit_depth,
            num_heads=cfg.vit_heads,
            num_labels=len(LABEL_NAMES),
        )
    else:
        model = TinyCNNDiagnoser(num_labels=len(LABEL_NAMES), width=cfg.cnn_width)

    model.to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # ---------------- opt ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val = 1e9
    best_path = os.path.join(cfg.out_dir, "diagnoser_best.pth")
    last_path = os.path.join(cfg.out_dir, "diagnoser_last.pth")

    print("\n[Train] start")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, loader_tr, opt, scaler, device, cfg.use_amp, thr=cfg.thr)
        va = eval_one_epoch(model, loader_va, device, cfg.use_amp, thr=cfg.thr)

        print(f"\n[Epoch {epoch:03d}/{cfg.epochs}] "
              f"train: L={tr['loss']:.4f} acc={tr['acc_macro']:.3f} ex={tr['exact_match']:.3f} | "
              f"val: L={va['loss']:.4f} acc={va.get('acc_macro',0):.3f} ex={va.get('exact_match',0):.3f} | "
              f"time={format_time(time.time()-t0)}")

        ckpt = {
            "epoch": epoch,
            "cfg": vars(cfg),
            "model": cfg.model,
            "label_names": LABEL_NAMES,
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
python scripts/train_diagnoser.py `
  --data_root "E:/ReAct-IR/data" `
  --out_dir "E:/ReAct-IR/checkpoints/diagnoser" `
  --split_train train --split_val test `
  --model vit `
  --epochs 10 --batch_size 16 --num_workers 4

"""