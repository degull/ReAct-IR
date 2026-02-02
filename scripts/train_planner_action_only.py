# scripts/train_planner_action_only.py
import os
import json
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm


# ---------------------------
# Action space (fixed order)
# ---------------------------
ACTIONS = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
A2I = {a: i for i, a in enumerate(ACTIONS)}
I2A = {i: a for a, i in A2I.items()}


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def choose_best_in_action(
    sample: Dict[str, Any],
    action: str,
    score_mode: str = "psnr",
    alpha_ssim: float = 50.0,
) -> Tuple[float, float]:
    """
    Returns:
      best_score, best_scale
    score_mode:
      - "psnr": maximize d_psnr
      - "combo": maximize d_psnr + alpha_ssim * d_ssim
    """
    sweep = sample.get("sweep", [])
    best_score = -1e18
    best_scale = 0.0
    for e in sweep:
        if e.get("action") != action:
            continue
        d_psnr = float(e.get("d_psnr", 0.0))
        d_ssim = float(e.get("d_ssim", 0.0))
        if score_mode == "combo":
            score = d_psnr + alpha_ssim * d_ssim
        else:
            score = d_psnr
        if score > best_score:
            best_score = score
            best_scale = float(e.get("scale", 0.0))
    return best_score, best_scale


def oracle_best_score(sample: Dict[str, Any], score_mode: str = "psnr", alpha_ssim: float = 50.0) -> float:
    b = sample.get("best", {})
    # "best" already contains best among all sweeps in your generator.
    d_psnr = float(b.get("d_psnr", 0.0))
    d_ssim = float(b.get("d_ssim", 0.0))
    if score_mode == "combo":
        return d_psnr + alpha_ssim * d_ssim
    return d_psnr


class RolloutsActionDataset(Dataset):
    """
    Action-only Planner dataset.
    x := [s0(5), m0_mean(5), m0_max(5), baseline_psnr, baseline_ssim] => 17 dims
    y := best.action (0..4)
    """

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        s0 = safe_get(it, ["state", "s0"], default=[0, 0, 0, 0, 0])
        m0m = safe_get(it, ["state", "m0_mean"], default=[0, 0, 0, 0, 0])
        m0x = safe_get(it, ["state", "m0_max"], default=[0, 0, 0, 0, 0])
        bpsnr = float(safe_get(it, ["baseline", "psnr_in"], default=0.0))
        bssim = float(safe_get(it, ["baseline", "ssim_in"], default=0.0))

        # robust casting + clipping (avoid NaNs)
        feat = np.array(list(s0) + list(m0m) + list(m0x) + [bpsnr, bssim], dtype=np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        best_action = safe_get(it, ["best", "action"], default=None)
        if best_action not in A2I:
            # fallback: meta.action or source_action
            best_action = safe_get(it, ["meta", "action"], default=None)
        if best_action not in A2I:
            best_action = safe_get(it, ["meta", "source_action"], default=None)
        if best_action not in A2I:
            best_action = "A_DEHAZE"  # last resort (should not happen)

        y = A2I[best_action]
        return torch.from_numpy(feat), torch.tensor(y, dtype=torch.long), it


class PlannerActionNet(nn.Module):
    def __init__(self, in_dim: int = 17, hidden: int = 256, depth: int = 3, dropout: float = 0.1, num_actions: int = 5):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_actions)

        # small init stabilization
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)  # [B,5]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_mode: str = "psnr",
    alpha_ssim: float = 50.0,
):
    model.eval()
    total = 0
    correct_action = 0
    loss_sum = 0.0

    # regret in terms of score_mode
    regret_sum = 0.0
    regret_abs_sum = 0.0

    # also track psnr regret specifically (human-friendly)
    psnr_regret_sum = 0.0

    ce = nn.CrossEntropyLoss(reduction="sum")

    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item())

        pred = torch.argmax(logits, dim=1)
        correct_action += int((pred == y).sum().item())
        total += int(y.numel())

        # regret (oracle best - best within predicted action)
        for bi in range(len(meta)):
            sample = meta[bi]
            pred_action = I2A[int(pred[bi].item())]

            pred_best_score, _ = choose_best_in_action(sample, pred_action, score_mode=score_mode, alpha_ssim=alpha_ssim)
            oracle_score = oracle_best_score(sample, score_mode=score_mode, alpha_ssim=alpha_ssim)
            r = float(oracle_score - pred_best_score)
            regret_sum += r
            regret_abs_sum += abs(r)

            # psnr-regret regardless of score mode:
            pred_best_psnr, _ = choose_best_in_action(sample, pred_action, score_mode="psnr", alpha_ssim=alpha_ssim)
            oracle_psnr = oracle_best_score(sample, score_mode="psnr", alpha_ssim=alpha_ssim)
            psnr_regret_sum += float(oracle_psnr - pred_best_psnr)

    avg_loss = loss_sum / max(1, total)
    acc = correct_action / max(1, total)
    avg_regret = regret_sum / max(1, total)
    avg_abs_regret = regret_abs_sum / max(1, total)
    avg_psnr_regret = psnr_regret_sum / max(1, total)

    return {
        "loss": avg_loss,
        "acc": acc,
        "avg_regret": avg_regret,
        "avg_abs_regret": avg_abs_regret,
        "avg_psnr_regret": avg_psnr_regret,
        "n": total,
    }

def collate_with_meta(batch):
    xs, ys, metas = [], [], []
    for x, y, meta in batch:
        xs.append(x)
        ys.append(y)
        metas.append(meta)
    return (
        torch.stack(xs, dim=0),
        torch.stack(ys, dim=0),
        metas,   # ← list[dict], 그대로 유지
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts_train", type=str, default="", help="train jsonl (recommended)")
    parser.add_argument("--rollouts_val", type=str, default="", help="val jsonl (optional)")
    parser.add_argument("--rollouts", type=str, default="", help="single jsonl -> auto split (if train/val not given)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    # evaluation regret mode
    parser.add_argument("--score_mode", type=str, default="psnr", choices=["psnr", "combo"])
    parser.add_argument("--alpha_ssim", type=float, default=50.0)

    # optional init ckpt
    parser.add_argument("--init_ckpt", type=str, default="")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---------------------------
    # Load rollouts
    # ---------------------------
    if args.rollouts_train:
        train_items = read_jsonl(args.rollouts_train)
        if not args.rollouts_val:
            raise ValueError("If --rollouts_train is given, please also provide --rollouts_val (recommended).")
        val_items = read_jsonl(args.rollouts_val)
    else:
        if not args.rollouts:
            raise ValueError("Provide either (--rollouts_train & --rollouts_val) or (--rollouts).")
        all_items = read_jsonl(args.rollouts)
        n = len(all_items)
        idx = list(range(n))
        random.shuffle(idx)
        nv = max(1, int(n * args.val_ratio))
        val_idx = set(idx[:nv])
        train_items = [all_items[i] for i in range(n) if i not in val_idx]
        val_items = [all_items[i] for i in range(n) if i in val_idx]

    print(f"[Data] train={len(train_items)} val={len(val_items)}")

    train_ds = RolloutsActionDataset(train_items)
    val_ds = RolloutsActionDataset(val_items)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta
    )


    # ---------------------------
    # Model
    # ---------------------------
    model = PlannerActionNet(
        in_dim=17, hidden=args.hidden, depth=args.depth, dropout=args.dropout, num_actions=len(ACTIONS)
    ).to(device)

    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Init] loaded strict=False  missing={len(missing)} unexpected={len(unexpected)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=bool(args.use_amp))

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "planner_action_best.pth")
    last_path = os.path.join(args.out_dir, "planner_action_last.pth")

    # ---------------------------
    # Train
    # ---------------------------
    print("[Train] start")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        loss_meter = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, ncols=120)
        for x, y, _meta in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=bool(args.use_amp)):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter += float(loss.item()) * int(y.numel())
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

            pbar.set_postfix({
                "L": f"{(loss_meter/max(1,total)):.4f}",
                "acc": f"{(correct/max(1,total)):.3f}",
            })

        train_loss = loss_meter / max(1, total)
        train_acc = correct / max(1, total)

        val_stats = evaluate(
            model, val_loader, device=device,
            score_mode=args.score_mode, alpha_ssim=args.alpha_ssim
        )
        val_loss = val_stats["loss"]
        val_acc = val_stats["acc"]
        val_regret = val_stats["avg_regret"]
        val_psnr_regret = val_stats["avg_psnr_regret"]

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.6f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.6f} acc={val_acc:.4f} "
            f"regret={val_regret:.4f} psnr_regret={val_psnr_regret:.4f} | "
            f"time={dt:.1f}s"
        )

        # save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "args": vars(args),
            "actions": ACTIONS,
        }, last_path)

        # save best by val_loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "args": vars(args),
                "actions": ACTIONS,
            }, best_path)
            print(f"[Save] BEST -> {best_path} (val_loss={best_val:.6f})")

    print("[Done]")


if __name__ == "__main__":
    main()
