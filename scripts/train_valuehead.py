# scripts/train_valuehead.py
import os
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, DefaultDict
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


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


def _fix_len(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == L:
        return x.astype(np.float32)
    y = np.zeros((L,), dtype=np.float32)
    n = min(L, x.shape[0])
    y[:n] = x[:n]
    return y.astype(np.float32)


def build_state_feat(state: Dict[str, Any]) -> np.ndarray:
    """
    state:
      s0: [5]
      m0_mean: [5]
      m0_max: [5]
    -> 15-d float32
    """
    s0 = _fix_len(np.asarray(state.get("s0", [0]*5), dtype=np.float32), 5)
    m0_mean = _fix_len(np.asarray(state.get("m0_mean", [0]*5), dtype=np.float32), 5)
    m0_max = _fix_len(np.asarray(state.get("m0_max", [0]*5), dtype=np.float32), 5)
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
# Dataset: expand (rollout item) x (action,scale)
# + keep group_id so we can do ranking within same state
# ------------------------------
class ValueRolloutDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        mode: str = "all_sweep",  # "all_sweep" or "best_only"
        value_clip_psnr: float = 50.0,
        value_clip_ssim: float = 1.0,
        scalar_w_ssim: float = 100.0,  # used for ranking ground-truth scalar
    ):
        """
        Each returned sample:
          x: state_feat(15) + action_onehot(5) + scale(1) => 21-d
          y: [d_psnr, d_ssim]
          meta: action_id, group_id, gt_scalar (for ranking)
        """
        assert mode in ["all_sweep", "best_only"]
        self.mode = mode
        self.value_clip_psnr = float(value_clip_psnr)
        self.value_clip_ssim = float(value_clip_ssim)
        self.scalar_w_ssim = float(scalar_w_ssim)

        self.samples: List[Tuple[np.ndarray, np.ndarray, int, int, float]] = []
        # (x, y, action_id, group_id, gt_scalar)

        group_id = 0
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
                        x, aid = self._make_x(sfeat, a, float(sc))
                        y = self._make_y(float(dps), float(dss))
                        gt_scalar = float(y[0] + self.scalar_w_ssim * y[1])
                        self.samples.append((x, y, aid, group_id, gt_scalar))
                        group_id += 1
            else:
                sweep = parse_sweep_entries(it)
                valid_cnt = 0
                for e in sweep:
                    a = e.get("action", None)
                    sc = e.get("scale", None)
                    dps = e.get("d_psnr", None)
                    dss = e.get("d_ssim", None)
                    if (a in ACTION_TO_ID) and (sc is not None) and (dps is not None) and (dss is not None):
                        x, aid = self._make_x(sfeat, a, float(sc))
                        y = self._make_y(float(dps), float(dss))
                        gt_scalar = float(y[0] + self.scalar_w_ssim * y[1])
                        self.samples.append((x, y, aid, group_id, gt_scalar))
                        valid_cnt += 1
                # group only meaningful if it had 2+ candidates
                if valid_cnt >= 2:
                    group_id += 1

        # if group_id ended up 0, still safe.

    def _make_x(self, sfeat: np.ndarray, action: str, scale: float) -> Tuple[np.ndarray, int]:
        aid = ACTION_TO_ID[action]
        aoh = one_hot(aid, len(ACTIONS))
        sc = np.asarray([scale], dtype=np.float32)
        x = np.concatenate([sfeat, aoh, sc], axis=0).astype(np.float32)  # (21,)
        return x, int(aid)

    def _make_y(self, dpsnr: float, dssim: float) -> np.ndarray:
        # clip (stability)
        dpsnr = float(np.clip(dpsnr, -self.value_clip_psnr, self.value_clip_psnr))
        dssim = float(np.clip(dssim, -self.value_clip_ssim, self.value_clip_ssim))
        return np.asarray([dpsnr, dssim], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, aid, gid, gts = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(aid, dtype=torch.long),
            torch.tensor(gid, dtype=torch.long),
            torch.tensor(gts, dtype=torch.float32),
        )


# ------------------------------
# Model (unchanged I/O: output is still 2-d)
# ------------------------------
class ValueHeadMLP(nn.Module):
    def __init__(self, in_dim: int = 21, hidden: int = 256, depth: int = 3, dropout: float = 0.1, out_dim: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------
# Loss helpers
# ------------------------------
def pred_scalar_from_pred2(pred2: torch.Tensor, w_ssim: float) -> torch.Tensor:
    # pred2: [B,2] -> scalar [B]
    return pred2[:, 0] + float(w_ssim) * pred2[:, 1]


def make_pairwise_ranking_loss(
    pred_scalar: torch.Tensor,  # [B]
    gt_scalar: torch.Tensor,    # [B]
    group_id: torch.Tensor,     # [B]
    margin: float = 0.0,
    max_pairs_per_group: int = 8,
) -> torch.Tensor:
    """
    For each group (same rollout state), sample pairs and enforce ordering.
    Use logistic/softplus ranking: softplus(-(s_i - s_j)*sign) with optional margin.
    sign = +1 if gt_i > gt_j else -1.
    """
    device = pred_scalar.device
    uniq = torch.unique(group_id)

    losses = []
    for g in uniq.tolist():
        idx = torch.nonzero(group_id == g, as_tuple=False).reshape(-1)
        if idx.numel() < 2:
            continue

        # random pair sampling inside group
        idx_list = idx.tolist()
        random.shuffle(idx_list)

        # build pairs
        pairs = []
        # simple: pair consecutive after shuffle (covers diverse pairs)
        for k in range(0, len(idx_list) - 1, 2):
            pairs.append((idx_list[k], idx_list[k + 1]))
            if len(pairs) >= max_pairs_per_group:
                break

        for i, j in pairs:
            gi = gt_scalar[i]
            gj = gt_scalar[j]
            if torch.isclose(gi, gj):
                continue

            si = pred_scalar[i]
            sj = pred_scalar[j]

            # want: (si - sj) has same sign as (gi - gj)
            sign = torch.sign(gi - gj)  # +1 or -1
            # logistic ranking with margin
            # loss = softplus( -( (si - sj) - margin ) * sign )
            diff = (si - sj) * sign
            loss_ij = F.softplus(-(diff - float(margin)))
            losses.append(loss_ij)

    if len(losses) == 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(losses).mean()


# ------------------------------
# Train/Eval
# ------------------------------
@torch.no_grad()
def eval_one_epoch(model, loader, device, scalar_w_ssim: float):
    model.eval()
    total = 0
    loss_sum = 0.0
    mae_psnr = 0.0
    mae_ssim = 0.0
    for x, y, _aid, _gid, _gts in loader:
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip: float = 1.0,
    scalar_w_ssim: float = 100.0,
    reg_weight: float = 1.0,
    rank_weight: float = 1.0,
    rank_margin: float = 0.0,
    rank_pairs_per_group: int = 8,
):
    model.train()
    total = 0
    loss_sum = 0.0
    reg_sum = 0.0
    rank_sum = 0.0

    for x, y, _aid, gid, gt_scalar in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        gid = gid.to(device, non_blocking=True)
        gt_scalar = gt_scalar.to(device, non_blocking=True)

        pred2 = model(x)  # [B,2]
        reg_loss = F.smooth_l1_loss(pred2, y)

        pred_scalar = pred_scalar_from_pred2(pred2, scalar_w_ssim)
        rank_loss = make_pairwise_ranking_loss(
            pred_scalar=pred_scalar,
            gt_scalar=gt_scalar,
            group_id=gid,
            margin=rank_margin,
            max_pairs_per_group=rank_pairs_per_group,
        )

        loss = float(reg_weight) * reg_loss + float(rank_weight) * rank_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        total += bs
        loss_sum += float(loss.item()) * bs
        reg_sum += float(reg_loss.item()) * bs
        rank_sum += float(rank_loss.item()) * bs

    if total == 0:
        return {"loss": 0.0, "reg_loss": 0.0, "rank_loss": 0.0}
    return {"loss": loss_sum / total, "reg_loss": reg_sum / total, "rank_loss": rank_sum / total}


def build_splits(items: List[Dict[str, Any]], val_ratio: float, seed: int):
    idxs = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = int(round(len(items) * val_ratio))
    val_ids = set(idxs[:n_val])
    train_items = [items[i] for i in idxs if i not in val_ids]
    val_items = [items[i] for i in idxs if i in val_ids]
    return train_items, val_items


def make_action_balanced_sampler(ds: ValueRolloutDataset) -> WeightedRandomSampler:
    # Count action frequency in expanded samples
    counts = np.zeros((len(ACTIONS),), dtype=np.float64)
    action_ids = []
    for _x, _y, aid, _gid, _gts in ds.samples:
        counts[int(aid)] += 1.0
        action_ids.append(int(aid))

    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts  # inverse freq
    weights = [float(inv[aid]) for aid in action_ids]
    weights_t = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_t, num_samples=len(weights), replacement=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=str, required=True,
                        help="e.g., E:/ReAct-IR/rollouts/rollouts_train.scale0base.jsonl")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all_sweep", choices=["all_sweep", "best_only"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--epochs", type=int, default=30)
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

    # --- ranking / scalarization ---
    parser.add_argument("--scalar_w_ssim", type=float, default=100.0,
                        help="scalar = d_psnr + scalar_w_ssim * d_ssim (used for ranking loss)")
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=0.0)
    parser.add_argument("--rank_pairs_per_group", type=int, default=8)

    # --- sampling ---
    parser.add_argument("--balance_actions", type=int, default=1,
                        help="1: use action-balanced sampler to reduce bias (recommended)")

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
        tr_items,
        mode=args.mode,
        value_clip_psnr=args.value_clip_psnr,
        value_clip_ssim=args.value_clip_ssim,
        scalar_w_ssim=args.scalar_w_ssim,
    )
    ds_va = ValueRolloutDataset(
        va_items,
        mode=args.mode,
        value_clip_psnr=args.value_clip_psnr,
        value_clip_ssim=args.value_clip_ssim,
        scalar_w_ssim=args.scalar_w_ssim,
    )
    print(f"[Dataset] train_samples={len(ds_tr)} val_samples={len(ds_va)} (expanded by sweep)")

    # Balanced sampler (helps when one action dominates in rollouts)
    sampler = None
    shuffle = True
    if int(args.balance_actions) == 1 and len(ds_tr) > 0:
        sampler = make_action_balanced_sampler(ds_tr)
        shuffle = False
        print("[Sampler] action-balanced sampler enabled (shuffle=False)")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

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

        tr = train_one_epoch(
            model,
            dl_tr,
            optim,
            device,
            grad_clip=args.grad_clip,
            scalar_w_ssim=args.scalar_w_ssim,
            reg_weight=args.reg_weight,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            rank_pairs_per_group=args.rank_pairs_per_group,
        )
        va = eval_one_epoch(model, dl_va, device, scalar_w_ssim=args.scalar_w_ssim)

        history["train"].append({"epoch": ep, **tr})
        history["val"].append({"epoch": ep, **va})

        print(
            f"[Epoch {ep:03d}/{args.epochs}] "
            f"train_loss={tr['loss']:.6f} (reg={tr['reg_loss']:.6f}, rank={tr['rank_loss']:.6f}) | "
            f"val_loss={va['loss']:.6f} mae_psnr={va['mae_psnr']:.4f} mae_ssim={va['mae_ssim']:.4f} | "
            f"time={time.time()-te0:.1f}s"
        )

        # save last
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "model_state_dict": model.state_dict(),  # extra key for compatibility
            "optim": optim.state_dict(),
            "args": vars(args),
            "history": history,
        }, ckpt_last)

        # save best (by val regression loss; stable)
        if va["loss"] < best_loss:
            best_loss = va["loss"]
            torch.save({
                "epoch": ep,
                "best_loss": best_loss,
                "model": model.state_dict(),
                "model_state_dict": model.state_dict(),
                "optim": optim.state_dict(),
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
