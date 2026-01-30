# scripts/train_planner.py
import os
import json
import time
import math
import random
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Constants
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


def build_state_feat(state: Dict[str, Any]) -> np.ndarray:
    s0 = np.asarray(state.get("s0", [0]*5), dtype=np.float32)
    m0_mean = np.asarray(state.get("m0_mean", [0]*5), dtype=np.float32)
    m0_max = np.asarray(state.get("m0_max", [0]*5), dtype=np.float32)

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
    return np.concatenate([s0, m0_mean, m0_max], axis=0).astype(np.float32)  # 15


def combo_id(action: str, scale: float, scales: List[float]) -> int:
    if action not in ACTION_TO_ID:
        return -1
    try:
        si = scales.index(float(scale))
    except ValueError:
        return -1
    return ACTION_TO_ID[action] * len(scales) + si


def decode_combo(cid: int, scales: List[float]) -> Tuple[str, float]:
    a = cid // len(scales)
    s = cid % len(scales)
    return ACTIONS[a], scales[s]


# ------------------------------
# ValueHead (teacher) loader
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


def make_valuehead_input(state15: torch.Tensor, action_id: int, scale: float) -> torch.Tensor:
    # state15: (B,15)
    B = state15.shape[0]
    aoh = torch.zeros((B, len(ACTIONS)), device=state15.device, dtype=state15.dtype)
    aoh[:, action_id] = 1.0
    sc = torch.full((B, 1), float(scale), device=state15.device, dtype=state15.dtype)
    return torch.cat([state15, aoh, sc], dim=1)  # (B,21)


@torch.no_grad()
def teacher_logits_from_valuehead(
    value_model: nn.Module,
    state15: torch.Tensor,
    scales: List[float],
    temp: float = 1.0,
    use_dpsnr_only: bool = True,
) -> torch.Tensor:
    """
    returns soft logits over combos: (B, num_actions*len(scales))
    """
    value_model.eval()
    B = state15.shape[0]
    C = len(ACTIONS) * len(scales)
    scores = torch.zeros((B, C), device=state15.device, dtype=state15.dtype)

    k = 0
    for a_id in range(len(ACTIONS)):
        for sc in scales:
            x = make_valuehead_input(state15, a_id, sc)
            pred = value_model(x)  # (B,2): [d_psnr, d_ssim]
            if use_dpsnr_only:
                v = pred[:, 0]
            else:
                # 간단 합성: d_psnr + 10*d_ssim (스케일 맞춤)
                v = pred[:, 0] + 10.0 * pred[:, 1]
            scores[:, k] = v
            k += 1

    # temperature-softmax용 logits 형태로 반환
    return scores / max(1e-6, float(temp))


# ------------------------------
# Planner Dataset
# ------------------------------
class PlannerDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], scales: List[float]):
        self.scales = [float(s) for s in scales]
        self.samples: List[Tuple[np.ndarray, int]] = []

        for it in items:
            state = it.get("state", {})
            if not isinstance(state, dict):
                continue
            sfeat = build_state_feat(state)

            b = it.get("best", None)
            if not isinstance(b, dict):
                continue
            a = b.get("action", None)
            sc = b.get("scale", None)
            if a not in ACTION_TO_ID or sc is None:
                continue
            cid = combo_id(a, float(sc), self.scales)
            if cid < 0:
                continue

            self.samples.append((sfeat, cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ------------------------------
# Planner Model
# ------------------------------
class PlannerMLP(nn.Module):
    def __init__(self, in_dim: int = 15, hidden: int = 256, depth: int = 3, dropout: float = 0.1, out_dim: int = 25):
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
# Train/Eval
# ------------------------------
@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    if total == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": loss_sum / total, "acc": correct / total}


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip: float,
    teacher: Optional[nn.Module],
    scales: List[float],
    kd_lambda: float,
    kd_temp: float,
):
    """
    기본: CE(logits, hard_label)
    선택: teacher(valuehead)로 distill 추가
      loss = CE + kd_lambda * KL(student || teacher)
    """
    model.train()
    total = 0
    loss_sum = 0.0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss_ce = F.cross_entropy(logits, y)

        loss = loss_ce

        if teacher is not None and kd_lambda > 0:
            with torch.no_grad():
                tlogits = teacher_logits_from_valuehead(
                    teacher, x, scales=scales, temp=kd_temp, use_dpsnr_only=True
                )  # (B,C)
                tprob = F.softmax(tlogits, dim=1)

            slog = F.log_softmax(logits / max(1e-6, float(kd_temp)), dim=1)
            # KL(teacher || student) 형태: sum tprob * (log(tprob)-log(sprob))
            loss_kd = F.kl_div(slog, tprob, reduction="batchmean")
            loss = loss_ce + float(kd_lambda) * loss_kd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return {"loss": (loss_sum / total) if total else 0.0, "acc": (correct / total) if total else 0.0}


def build_splits(items: List[Dict[str, Any]], val_ratio: float, seed: int):
    idxs = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = int(round(len(items) * val_ratio))
    val_ids = set(idxs[:n_val])
    tr = [items[i] for i in idxs if i not in val_ids]
    va = [items[i] for i in idxs if i in val_ids]
    return tr, va


def load_valuehead_teacher(ckpt_path: str, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    hidden = int(args.get("hidden", 256))
    depth = int(args.get("depth", 3))
    dropout = float(args.get("dropout", 0.1))
    model = ValueHeadMLP(in_dim=21, hidden=hidden, depth=depth, dropout=dropout, out_dim=2)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str, required=True, help="scale0base jsonl recommended")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # teacher distill
    p.add_argument("--valuehead_ckpt", type=str, default="", help="optional: E:/.../valuehead_best.pth")
    p.add_argument("--kd_lambda", type=float, default=0.0, help="0이면 distill off")
    p.add_argument("--kd_temp", type=float, default=1.0)

    args = p.parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Load] {args.rollouts}")
    items = load_jsonl(args.rollouts)
    print(f"[Load] items={len(items)}")

    scales = [float(s) for s in args.scales]
    num_classes = len(ACTIONS) * len(scales)
    print(f"[Config] scales={scales} -> num_classes={num_classes}")

    tr_items, va_items = build_splits(items, args.val_ratio, args.seed)
    ds_tr = PlannerDataset(tr_items, scales=scales)
    ds_va = PlannerDataset(va_items, scales=scales)
    print(f"[Dataset] train={len(ds_tr)} val={len(ds_va)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = PlannerMLP(in_dim=15, hidden=args.hidden, depth=args.depth, dropout=args.dropout, out_dim=num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    teacher = None
    if args.valuehead_ckpt and os.path.isfile(args.valuehead_ckpt) and args.kd_lambda > 0:
        teacher = load_valuehead_teacher(args.valuehead_ckpt, device=device)
        print(f"[Teacher] loaded: {args.valuehead_ckpt} (kd_lambda={args.kd_lambda}, kd_temp={args.kd_temp})")
    else:
        print("[Teacher] off")

    best_loss = float("inf")
    history = {"train": [], "val": []}

    ckpt_best = os.path.join(args.out_dir, "planner_best.pth")
    ckpt_last = os.path.join(args.out_dir, "planner_last.pth")
    info_path = os.path.join(args.out_dir, "planner_train_info.json")

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        te0 = time.time()
        tr = train_one_epoch(
            model, dl_tr, optim, device,
            grad_clip=args.grad_clip,
            teacher=teacher,
            scales=scales,
            kd_lambda=args.kd_lambda,
            kd_temp=args.kd_temp,
        )
        va = eval_one_epoch(model, dl_va, device)

        history["train"].append({"epoch": ep, **tr})
        history["val"].append({"epoch": ep, **va})

        print(
            f"[Epoch {ep:03d}/{args.epochs}] "
            f"train_loss={tr['loss']:.6f} acc={tr['acc']:.4f} | "
            f"val_loss={va['loss']:.6f} acc={va['acc']:.4f} | "
            f"time={time.time()-te0:.1f}s"
        )

        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "args": vars(args),
            "scales": scales,
            "history": history,
        }, ckpt_last)

        if va["loss"] < best_loss:
            best_loss = va["loss"]
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "scales": scales,
                "history": history,
            }, ckpt_best)
            print(f"[Save] BEST -> {ckpt_best} (val_loss={best_loss:.6f})")

        save_json(info_path, {
            "rollouts": args.rollouts,
            "out_dir": args.out_dir,
            "device": device,
            "num_classes": num_classes,
            "scales": scales,
            "best_loss": best_loss,
            "time_elapsed_sec": time.time() - t0,
            "history": history,
        })

    print("[Done]")


if __name__ == "__main__":
    main()


"""
python scripts/train_planner.py `
  --rollouts "E:/ReAct-IR/rollouts/rollouts_train.scale0base.jsonl" `
  --out_dir  "E:/ReAct-IR/checkpoints/planner" `
  --scales 0 0.25 0.5 0.75 1.0 `
  --epochs 30 --batch_size 256 --num_workers 4

"""

"""
python scripts/train_planner.py `
  --rollouts "E:/ReAct-IR/rollouts/rollouts_train.scale0base.jsonl" `
  --out_dir  "E:/ReAct-IR/checkpoints/planner_kd" `
  --scales 0 0.25 0.5 0.75 1.0 `
  --valuehead_ckpt "E:/ReAct-IR/checkpoints/valuehead/valuehead_best.pth" `
  --kd_lambda 0.5 --kd_temp 1.5 `
  --epochs 30 --batch_size 256 --num_workers 4

"""