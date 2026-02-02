# scripts/eval_planner.py
import os
import json
import argparse
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


ACTIONS = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
A2I = {a: i for i, a in enumerate(ACTIONS)}
I2A = {i: a for a, i in A2I.items()}


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


def build_feat(sample: Dict[str, Any]) -> np.ndarray:
    s0 = safe_get(sample, ["state", "s0"], default=[0, 0, 0, 0, 0])
    m0m = safe_get(sample, ["state", "m0_mean"], default=[0, 0, 0, 0, 0])
    m0x = safe_get(sample, ["state", "m0_max"], default=[0, 0, 0, 0, 0])
    bpsnr = float(safe_get(sample, ["baseline", "psnr_in"], default=0.0))
    bssim = float(safe_get(sample, ["baseline", "ssim_in"], default=0.0))
    feat = np.array(list(s0) + list(m0m) + list(m0x) + [bpsnr, bssim], dtype=np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def choose_best_in_action(
    sample: Dict[str, Any],
    action: str,
    score_mode: str = "psnr",
    alpha_ssim: float = 50.0,
) -> Tuple[float, float, float, float]:
    """
    Returns:
      (best_score, best_scale, best_d_psnr, best_d_ssim)
    """
    sweep = sample.get("sweep", [])
    best_score = -1e18
    best_scale = 0.0
    best_dpsnr = 0.0
    best_dssim = 0.0

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
            best_dpsnr = d_psnr
            best_dssim = d_ssim

    return best_score, best_scale, best_dpsnr, best_dssim


def oracle_best(sample: Dict[str, Any], score_mode: str = "psnr", alpha_ssim: float = 50.0) -> Tuple[str, float, float, float]:
    b = sample.get("best", {})
    act = b.get("action", None)
    d_psnr = float(b.get("d_psnr", 0.0))
    d_ssim = float(b.get("d_ssim", 0.0))
    if score_mode == "combo":
        score = d_psnr + alpha_ssim * d_ssim
    else:
        score = d_psnr
    return act, score, d_psnr, d_ssim


class PlannerActionNet(nn.Module):
    def __init__(self, in_dim: int = 17, hidden: int = 256, depth: int = 3, dropout: float = 0.1, num_actions: int = 5):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


def confusion_to_text(mat: np.ndarray, labels: List[str]) -> str:
    # rows: gt, cols: pred
    colw = max(7, max(len(x) for x in labels) + 2)
    header = " " * colw + "".join([f"{l:>{colw}}" for l in labels])
    lines = [header]
    for i, l in enumerate(labels):
        row = f"{l:<{colw}}" + "".join([f"{int(mat[i, j]):>{colw}d}" for j in range(len(labels))])
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=str, required=True, help="jsonl rollouts")
    parser.add_argument("--planner_ckpt", type=str, required=True, help="planner_action_best.pth")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--score_mode", type=str, default="psnr", choices=["psnr", "combo"])
    parser.add_argument("--alpha_ssim", type=float, default=50.0)

    parser.add_argument("--max_print", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    items = read_jsonl(args.rollouts)
    print(f"[Load] {args.rollouts}")
    print(f"[Load] items = {len(items)}")

    # load ckpt
    ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    ck_args = ckpt.get("args", {})
    hidden = int(ck_args.get("hidden", 256))
    depth = int(ck_args.get("depth", 3))
    dropout = float(ck_args.get("dropout", 0.1))

    model = PlannerActionNet(in_dim=17, hidden=hidden, depth=depth, dropout=dropout, num_actions=len(ACTIONS)).to(device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # stats
    n = 0
    correct_action = 0

    conf = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)

    regret_score_sum = 0.0
    regret_psnr_sum = 0.0
    regret_ssim_sum = 0.0

    chosen_dpsnr_sum = 0.0
    chosen_dssim_sum = 0.0

    oracle_dpsnr_sum = 0.0
    oracle_dssim_sum = 0.0

    worst = []  # list of (regret_psnr, id, input_path, gt_action, pred_action, pred_scale)

    per_gt = defaultdict(lambda: {"n": 0, "acc": 0, "regret_psnr": 0.0})

    with torch.no_grad():
        for it in tqdm(items, ncols=120):
            feat = build_feat(it)
            x = torch.from_numpy(feat).unsqueeze(0).to(device)
            logits = model(x)
            pred_i = int(torch.argmax(logits, dim=1).item())
            pred_action = I2A[pred_i]

            gt_action = safe_get(it, ["best", "action"], default=None)
            if gt_action not in A2I:
                gt_action = safe_get(it, ["meta", "action"], default=None)
            if gt_action not in A2I:
                gt_action = safe_get(it, ["meta", "source_action"], default=None)
            if gt_action not in A2I:
                gt_action = "A_DEHAZE"

            gt_i = A2I[gt_action]
            conf[gt_i, pred_i] += 1

            if pred_action == gt_action:
                correct_action += 1

            # pick best scale within predicted action
            pred_best_score, pred_scale, pred_dpsnr, pred_dssim = choose_best_in_action(
                it, pred_action, score_mode=args.score_mode, alpha_ssim=args.alpha_ssim
            )
            oracle_act, oracle_score, oracle_dpsnr, oracle_dssim = oracle_best(
                it, score_mode=args.score_mode, alpha_ssim=args.alpha_ssim
            )

            # regrets
            regret_score = float(oracle_score - pred_best_score)
            regret_score_sum += regret_score

            regret_psnr = float(oracle_dpsnr - pred_dpsnr)
            regret_ssim = float(oracle_dssim - pred_dssim)

            regret_psnr_sum += regret_psnr
            regret_ssim_sum += regret_ssim

            chosen_dpsnr_sum += pred_dpsnr
            chosen_dssim_sum += pred_dssim
            oracle_dpsnr_sum += oracle_dpsnr
            oracle_dssim_sum += oracle_dssim

            n += 1

            per_gt[gt_action]["n"] += 1
            per_gt[gt_action]["acc"] += int(pred_action == gt_action)
            per_gt[gt_action]["regret_psnr"] += regret_psnr

            # collect worst examples
            inp = it.get("input", safe_get(it, ["meta", "input_path"], default=""))
            sid = it.get("id", -1)
            worst.append((regret_psnr, sid, inp, gt_action, pred_action, pred_scale))

    acc = correct_action / max(1, n)
    avg_regret_score = regret_score_sum / max(1, n)
    avg_regret_psnr = regret_psnr_sum / max(1, n)
    avg_regret_ssim = regret_ssim_sum / max(1, n)

    avg_chosen_dpsnr = chosen_dpsnr_sum / max(1, n)
    avg_oracle_dpsnr = oracle_dpsnr_sum / max(1, n)

    print("\n==================== Planner Eval Report ====================")
    print(f"[Items] {n}")
    print(f"[Action acc] {acc:.4f}  (correct={correct_action}/{n})")
    print(f"[Regret:{args.score_mode}] avg={avg_regret_score:.4f}")
    print(f"[Regret PSNR] avg={avg_regret_psnr:.4f}")
    print(f"[Regret SSIM] avg={avg_regret_ssim:.6f}")
    print(f"[Chosen ΔPSNR] avg={avg_chosen_dpsnr:.4f} | [Oracle ΔPSNR] avg={avg_oracle_dpsnr:.4f}")

    # save confusion
    conf_txt = confusion_to_text(conf, ACTIONS)
    conf_path = os.path.join(args.out_dir, "confusion_matrix.txt")
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(conf_txt + "\n")
    print(f"[Save] confusion matrix -> {conf_path}")

    # per-action report
    per_path = os.path.join(args.out_dir, "per_action.txt")
    with open(per_path, "w", encoding="utf-8") as f:
        f.write("Per-GT-Action stats\n")
        for a in ACTIONS:
            nn = per_gt[a]["n"]
            if nn == 0:
                continue
            aa = per_gt[a]["acc"] / nn
            rr = per_gt[a]["regret_psnr"] / nn
            f.write(f"{a}: n={nn} acc={aa:.4f} avg_psnr_regret={rr:.4f}\n")
    print(f"[Save] per-action -> {per_path}")

    # worst samples
    worst.sort(key=lambda x: x[0], reverse=True)
    worst_path = os.path.join(args.out_dir, "worst_regret_psnr.jsonl")
    with open(worst_path, "w", encoding="utf-8") as f:
        for row in worst[: max(1, args.max_print)]:
            regret_psnr, sid, inp, gt_action, pred_action, pred_scale = row
            f.write(json.dumps({
                "regret_psnr": float(regret_psnr),
                "id": sid,
                "input": inp,
                "gt_action": gt_action,
                "pred_action": pred_action,
                "pred_scale": float(pred_scale),
            }, ensure_ascii=False) + "\n")
    print(f"[Save] worst samples -> {worst_path}")

    # summary json
    summary = {
        "items": n,
        "action_acc": acc,
        "avg_regret_score": avg_regret_score,
        "avg_regret_psnr": avg_regret_psnr,
        "avg_regret_ssim": avg_regret_ssim,
        "avg_chosen_dpsnr": avg_chosen_dpsnr,
        "avg_oracle_dpsnr": avg_oracle_dpsnr,
        "score_mode": args.score_mode,
        "alpha_ssim": args.alpha_ssim,
    }
    summ_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Save] summary -> {summ_path}")

    print("[Done]")


if __name__ == "__main__":
    main()
