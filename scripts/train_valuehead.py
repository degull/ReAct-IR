# scripts/train_valuehead.py
import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Utils
# ------------------------------
ACTIONS = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def build_state_feat(state: Dict[str, Any]) -> np.ndarray:
    """
    state:
      s0: [5]
      m0_mean: [5]
      m0_max: [5]
    -> 15-d float32
    """
    s0 = np.asarray(state.get("s0", [0]*5), dtype=np.float32)
    m0_mean = np.asarray(state.get("m0_mean", [0]*5), dtype=np.float32)
    m0_max = np.asarray(state.get("m0_max", [0]*5), dtype=np.float32)

    # 안정성: 길이 보정
    def fix_len(x, L=5):
        if x.shape[0] == L:
            return x
        y = np.zeros((L,), dtype=np.float32)
        n = min(L, x.shape[0])
        y[:n] = x[:n]
        return y

    s0 = fix_len(s0, 5)
    m0_mean = fix_len(m0_mean, 5)
    m0_max = fix_len(m0_max, 5)

    feat = np.concatenate([s0, m0_mean, m0_max], axis=0)  # (15,)
    return feat.astype(np.float32)


def parse_sweep_entries(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    sweep = item.get("sweep", [])
    if not isinstance(sweep, list):
        return []
    out = []
    for e in sweep:
        if not isinstance(e, dict):
            continue
        a = e.get("action", None)
        sc = e.get("scale", None)
        if a is None or sc is None:
            continue
        out.append(e)
    return out


# ------------------------------
# Dataset: expand (sample) x (action,scale)
# ------------------------------
class ValueRolloutDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        mode: str = "all_sweep",  # "all_sweep" or "best_only"
        value_clip_psnr: float = 50.0,
        value_clip_ssim: float = 1.0,
    ):
        """
        Each returned sample:
          x: state_feat(15) + action_onehot(5) + scale(1) => 21-d
          y: [d_psnr, d_ssim]
        """
        assert mode in ["all_sweep", "best_only"]
        self.mode = mode
        self.value_clip_psnr = float(value_clip_psnr)
        self.value_clip_ssim = float(value_clip_ssim)

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for it in items:
            state = it.get("state", {})
            if not isinstance(state, dict):
                continue
            sfeat = build_state_feat(state)

            if self.mode == "best_only":
                b = it.get("best", None)
                if isinstance(b, dict):
                    a = b.get("action", None)
                    sc = b.get("scale", None)
                    dps = b.get("d_psnr", None)
                    dss = b.get("d_ssim", None)
                    if (a in ACTION_TO_ID) and (sc is not None) and (dps is not None) and (dss is not None):
                        x = self._make_x(sfeat, a, float(sc))
                        y = self._make_y(float(dps), float(dss))
                        self.samples.append((x, y))
            else:
                sweep = parse_sweep_entries(it)
                for e in sweep:
                    a = e.get("action", None)
                    sc = e.get("scale", None)
                    dps = e.get("d_psnr", None)
                    dss = e.get("d_ssim", None)
                    if (a in ACTION_TO_ID) and (sc is not None) and (dps is not None) and (dss is not None):
                        x = self._make_x(sfeat, a, float(sc))
                        y = self._make_y(float(dps), float(dss))
                        self.samples.append((x, y))

    def _make_x(self, sfeat: np.ndarray, action: str, scale: float) -> np.ndarray:
        aid = ACTION_TO_ID[action]
        aoh = one_hot(aid, len(ACTIONS))
        sc = np.asarray([scale], dtype=np.float32)
        x = np.concatenate([sfeat, aoh, sc], axis=0).astype(np.float32)  # (21,)
        return x

    def _make_y(self, dpsnr: float, dssim: float) -> np.ndarray:
        # clip (안정성)
        dpsnr = float(np.clip(dpsnr, -self.value_clip_psnr, self.value_clip_psnr))
        dssim = float(np.clip(dssim, -self.value_clip_ssim, self.value_clip_ssim))
        return np.asarray([dpsnr, dssim], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------------------
# Model
# ------------------------------
class ValueHeadMLP(nn.Module):
    def __init__(self, in_dim: int = 21, hidden: int = 256, depth: int = 3, dropout: float = 0.1, out_dim: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------
# Train/Eval
# ------------------------------
@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total = 0
    loss_sum = 0.0
    mae_psnr = 0.0
    mae_ssim = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = F.smooth_l1_loss(pred, y)
        loss_sum += float(loss.item()) * x.size(0)
        total += x.size(0)
        mae = torch.abs(pred - y)
        mae_psnr += float(mae[:, 0].mean().item()) * x.size(0)
        mae_ssim += float(mae[:, 1].mean().item()) * x.size(0)
    if total == 0:
        return {"loss": 0.0, "mae_psnr": 0.0, "mae_ssim": 0.0}
    return {
        "loss": loss_sum / total,
        "mae_psnr": mae_psnr / total,
        "mae_ssim": mae_ssim / total,
    }


def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 1.0):
    model.train()
    total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = F.smooth_l1_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += float(loss.item()) * x.size(0)
        total += x.size(0)

    return {"loss": (loss_sum / total) if total > 0 else 0.0}


def build_splits(items: List[Dict[str, Any]], val_ratio: float, seed: int):
    idxs = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = int(round(len(items) * val_ratio))
    val_ids = set(idxs[:n_val])
    train_items = [items[i] for i in idxs if i not in val_ids]
    val_items = [items[i] for i in idxs if i in val_ids]
    return train_items, val_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=str, required=True, help="e.g., E:/ReAct-IR/rollouts/rollouts_train.scale0base.jsonl")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all_sweep", choices=["all_sweep", "best_only"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--value_clip_psnr", type=float, default=50.0)
    parser.add_argument("--value_clip_ssim", type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Load] {args.rollouts}")
    items = load_jsonl(args.rollouts)
    print(f"[Load] items={len(items)}")

    tr_items, va_items = build_splits(items, args.val_ratio, args.seed)
    print(f"[Split] train_items={len(tr_items)} val_items={len(va_items)} mode={args.mode}")

    ds_tr = ValueRolloutDataset(
        tr_items, mode=args.mode,
        value_clip_psnr=args.value_clip_psnr, value_clip_ssim=args.value_clip_ssim
    )
    ds_va = ValueRolloutDataset(
        va_items, mode=args.mode,
        value_clip_psnr=args.value_clip_psnr, value_clip_ssim=args.value_clip_ssim
    )
    print(f"[Dataset] train_samples={len(ds_tr)} val_samples={len(ds_va)}  (expanded by sweep)")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = ValueHeadMLP(in_dim=21, hidden=args.hidden, depth=args.depth, dropout=args.dropout, out_dim=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    history = {"train": [], "val": []}

    ckpt_best = os.path.join(args.out_dir, "valuehead_best.pth")
    ckpt_last = os.path.join(args.out_dir, "valuehead_last.pth")
    info_path = os.path.join(args.out_dir, "valuehead_train_info.json")

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        te0 = time.time()
        tr = train_one_epoch(model, dl_tr, optim, device, grad_clip=args.grad_clip)
        va = eval_one_epoch(model, dl_va, device)

        history["train"].append({"epoch": ep, **tr})
        history["val"].append({"epoch": ep, **va})

        print(
            f"[Epoch {ep:03d}/{args.epochs}] "
            f"train_loss={tr['loss']:.6f} | "
            f"val_loss={va['loss']:.6f} mae_psnr={va['mae_psnr']:.4f} mae_ssim={va['mae_ssim']:.4f} | "
            f"time={time.time()-te0:.1f}s"
        )

        # save last
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "args": vars(args),
            "history": history,
        }, ckpt_last)

        # save best
        if va["loss"] < best_loss:
            best_loss = va["loss"]
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "history": history,
            }, ckpt_best)
            print(f"[Save] BEST -> {ckpt_best} (val_loss={best_loss:.6f})")

        save_json(info_path, {
            "rollouts": args.rollouts,
            "out_dir": args.out_dir,
            "device": device,
            "best_loss": best_loss,
            "time_elapsed_sec": time.time() - t0,
            "history": history,
        })

    print("[Done]")


if __name__ == "__main__":
    main()
