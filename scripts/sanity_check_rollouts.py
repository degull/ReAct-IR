# scripts/sanity_check_rollouts.py
import os
import json
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional

ACTIONS = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP", "UNK"]

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def quantiles(xs: List[float], qs=(0, 10, 25, 50, 75, 90, 100)) -> Dict[str, float]:
    if not xs:
        return {f"q{q:02d}": float("nan") for q in qs}
    xs2 = sorted(xs)
    n = len(xs2)
    out = {}
    for q in qs:
        if n == 1:
            out[f"q{q:02d}"] = xs2[0]
            continue
        pos = (q / 100.0) * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out[f"q{q:02d}"] = xs2[lo]
        else:
            t = pos - lo
            out[f"q{q:02d}"] = xs2[lo] * (1 - t) + xs2[hi] * t
    return out

def stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), **quantiles([])}
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, (n - 1))
    s = math.sqrt(v)
    return {"n": n, "mean": m, "std": s, **quantiles(xs)}

def action_idx(a: str) -> int:
    if a in ACTIONS:
        return ACTIONS.index(a)
    return ACTIONS.index("UNK")

def get_baseline(item: Dict[str, Any], baseline_mode: str) -> Tuple[Optional[float], Optional[float]]:
    """
    baseline_mode:
      - "input": use baseline.psnr_in / baseline.ssim_in
      - "scale0": use baseline.psnr_base / baseline.ssim_base (rebased outputs)
    """
    b = item.get("baseline", {})
    if not isinstance(b, dict):
        return None, None

    if baseline_mode == "input":
        ps = b.get("psnr_in", None)
        ss = b.get("ssim_in", None)
    else:
        ps = b.get("psnr_base", None)
        ss = b.get("ssim_base", None)

    if ps is None or ss is None or (not is_finite(ps)) or (not is_finite(ss)):
        return None, None
    return float(ps), float(ss)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--max_print", default=50, type=int)

    ap.add_argument("--baseline_mode", default="scale0", choices=["scale0", "input"],
                    help="scale0(추천): baseline=backbone(scale0). input: baseline=input-vs-gt")
    ap.add_argument("--scale0", default=0.0, type=float, help="used for some checks (informational)")
    ap.add_argument("--mismatch_tol", default=1e-3, type=float)

    ap.add_argument("--huge_delta", default=10.0, type=float)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Load] {args.rollouts}")
    items = read_jsonl(args.rollouts)
    print(f"[Load] items = {len(items)}")

    # ---------------------------------------------------------
    # Collect stats
    # ---------------------------------------------------------
    abs_psnr0_minus_baseline = []
    abs_ssim0_minus_baseline = []
    range_psnr0_across_actions = []
    range_ssim0_across_actions = []

    dpsnr_recompute_abs_err = []
    dssim_recompute_abs_err = []

    # best vs source_action confusion
    conf = [[0 for _ in ACTIONS] for _ in ACTIONS]
    best_correct = 0
    best_total = 0

    # per-action stats (best)
    best_dpsnr_by_action = {a: [] for a in ACTIONS}
    sweep_dpsnr_by_action = {a: [] for a in ACTIONS}

    scale0_mismatch_samples = []
    scale0_cross_action_samples = []
    dpsnr_bad = []
    dssim_bad = []
    huge_delta = []

    for idx, it in enumerate(items, start=1):
        sweep = it.get("sweep", [])
        if not isinstance(sweep, list) or len(sweep) == 0:
            continue

        ps_base, ss_base = get_baseline(it, args.baseline_mode)
        if ps_base is None or ss_base is None:
            continue

        # gather scale0 stats (psnr0 compared to chosen baseline)
        ps0_list = []
        ss0_list = []
        for e in sweep:
            if float(e.get("scale", -999)) == float(args.scale0):
                ps0_list.append(float(e.get("psnr")))
                ss0_list.append(float(e.get("ssim")))

        # scale0 should exist for all actions typically; but be tolerant
        if ps0_list:
            # abs diff between scale0 and baseline
            # (if baseline_mode=scale0, this should be near 0)
            abs_psnr0_minus_baseline.append(abs(ps0_list[0] - ps_base))
            abs_ssim0_minus_baseline.append(abs(ss0_list[0] - ss_base))

            # cross-action range at scale0
            range_psnr0_across_actions.append(max(ps0_list) - min(ps0_list))
            range_ssim0_across_actions.append(max(ss0_list) - min(ss0_list))

            if abs(ps0_list[0] - ps_base) > args.mismatch_tol:
                scale0_mismatch_samples.append((abs(ps0_list[0] - ps_base), idx, it.get("input", "")))

            if (max(ps0_list) - min(ps0_list)) > args.mismatch_tol:
                scale0_cross_action_samples.append((max(ps0_list) - min(ps0_list), idx, it.get("input", "")))

        # recompute deltas for each sweep entry vs chosen baseline
        for e in sweep:
            ps = e.get("psnr", None)
            ss = e.get("ssim", None)
            if ps is None or ss is None or (not is_finite(ps)) or (not is_finite(ss)):
                continue
            ps = float(ps); ss = float(ss)
            dps = ps - ps_base
            dss = ss - ss_base

            if e.get("d_psnr", None) is not None and is_finite(e["d_psnr"]):
                dpsnr_recompute_abs_err.append(abs(float(e["d_psnr"]) - dps))
                if abs(float(e["d_psnr"]) - dps) > args.mismatch_tol:
                    dpsnr_bad.append((abs(float(e["d_psnr"]) - dps), idx, it.get("input", ""), e.get("action",""), e.get("scale", None)))
            if e.get("d_ssim", None) is not None and is_finite(e["d_ssim"]):
                dssim_recompute_abs_err.append(abs(float(e["d_ssim"]) - dss))
                if abs(float(e["d_ssim"]) - dss) > args.mismatch_tol:
                    dssim_bad.append((abs(float(e["d_ssim"]) - dss), idx, it.get("input", ""), e.get("action",""), e.get("scale", None)))

            # huge delta logging
            if abs(dps) > args.huge_delta:
                huge_delta.append((dps, idx, it.get("input",""), e.get("action",""), float(e.get("scale", -1))))

            # per action sweep stats
            a = e.get("action", "UNK")
            if a not in ACTIONS: a = "UNK"
            sweep_dpsnr_by_action[a].append(dps)

        # best confusion
        best = it.get("best", None)
        src = (it.get("meta", {}) or {}).get("source_action", (it.get("meta", {}) or {}).get("action", "UNK"))
        if not isinstance(src, str): src = "UNK"
        if src not in ACTIONS: src = "UNK"

        if isinstance(best, dict):
            ba = best.get("action", "UNK")
            if not isinstance(ba, str): ba = "UNK"
            if ba not in ACTIONS: ba = "UNK"
            conf[action_idx(src)][action_idx(ba)] += 1
            best_total += 1
            if ba == src:
                best_correct += 1

            # best delta
            bdps = best.get("d_psnr", None)
            if bdps is not None and is_finite(bdps):
                best_dpsnr_by_action[ba].append(float(bdps))

    # ---------------------------------------------------------
    # Print report
    # ---------------------------------------------------------
    print("\n==================== Sanity Report ====================")
    print(f"[Items] {len(items)}")
    print(f"[Baseline mode] {args.baseline_mode}  (scale0={args.scale0})")

    print("\n--- scale=0.0 vs baseline ---")
    s1 = stats(abs_psnr0_minus_baseline)
    s2 = stats(abs_ssim0_minus_baseline)
    print(f"[abs(psnr0 - baseline)] n={s1['n']}  mean={s1['mean']:.6f}  std={s1['std']:.6f}  "
          f"q00={s1['q00']:.6f}  q10={s1['q10']:.6f}  q25={s1['q25']:.6f}  q50={s1['q50']:.6f}  q75={s1['q75']:.6f}  q90={s1['q90']:.6f}  q100={s1['q100']:.6f}")
    print(f"[abs(ssim0 - baseline)] n={s2['n']}  mean={s2['mean']:.6f}  std={s2['std']:.6f}  "
          f"q00={s2['q00']:.6f}  q10={s2['q10']:.6f}  q25={s2['q25']:.6f}  q50={s2['q50']:.6f}  q75={s2['q75']:.6f}  q90={s2['q90']:.6f}  q100={s2['q100']:.6f}")

    print("\n--- scale=0.0 cross-action range ---")
    r1 = stats(range_psnr0_across_actions)
    r2 = stats(range_ssim0_across_actions)
    print(f"[range(psnr0 across actions)] n={r1['n']}  mean={r1['mean']:.6f}  std={r1['std']:.6f}  "
          f"q00={r1['q00']:.6f}  q10={r1['q10']:.6f}  q25={r1['q25']:.6f}  q50={r1['q50']:.6f}  q75={r1['q75']:.6f}  q90={r1['q90']:.6f}  q100={r1['q100']:.6f}")
    print(f"[range(ssim0 across actions)] n={r2['n']}  mean={r2['mean']:.6f}  std={r2['std']:.6f}  "
          f"q00={r2['q00']:.6f}  q10={r2['q10']:.6f}  q25={r2['q25']:.6f}  q50={r2['q50']:.6f}  q75={r2['q75']:.6f}  q90={r2['q90']:.6f}  q100={r2['q100']:.6f}")

    print("\n--- delta recompute errors (all sweep entries) ---")
    e1 = stats(dpsnr_recompute_abs_err)
    e2 = stats(dssim_recompute_abs_err)
    print(f"[abs(recomputed d_psnr - stored d_psnr)] n={e1['n']}  mean={e1['mean']:.6f}  std={e1['std']:.6f}  "
          f"q00={e1['q00']:.6f}  q10={e1['q10']:.6f}  q25={e1['q25']:.6f}  q50={e1['q50']:.6f}  q75={e1['q75']:.6f}  q90={e1['q90']:.6f}  q100={e1['q100']:.6f}")
    print(f"[abs(recomputed d_ssim - stored d_ssim)] n={e2['n']}  mean={e2['mean']:.6f}  std={e2['std']:.6f}  "
          f"q00={e2['q00']:.6f}  q10={e2['q10']:.6f}  q25={e2['q25']:.6f}  q50={e2['q50']:.6f}  q75={e2['q75']:.6f}  q90={e2['q90']:.6f}  q100={e2['q100']:.6f}")

    print("\n--- best-action vs source_action ---")
    acc = (best_correct / best_total) if best_total > 0 else float("nan")
    print(f"[Best Action] acc={acc:.4f}  (correct={best_correct} / total={best_total})")
    header = "            " + "  ".join([f"{a:>7s}" for a in ACTIONS])
    print(header)
    for i, ra in enumerate(ACTIONS):
        row = [conf[i][j] for j in range(len(ACTIONS))]
        print(f"{ra:>7s} " + " ".join([f"{v:7d}" for v in row]))

    print("\n--- per-action BEST ΔPSNR stats ---")
    for a in ACTIONS:
        xs = best_dpsnr_by_action[a]
        if not xs:
            continue
        st = stats(xs)
        print(f"[BEST {a}] n={st['n']} mean={st['mean']:.6f} std={st['std']:.6f} median={st['q50']:.6f} q10={st['q10']:.6f} q90={st['q90']:.6f}")

    print("\n--- per-action ALL-SWEEP ΔPSNR stats ---")
    for a in ACTIONS:
        xs = sweep_dpsnr_by_action[a]
        if not xs:
            continue
        st = stats(xs)
        print(f"[SWEEP {a}] n={st['n']} mean={st['mean']:.6f} std={st['std']:.6f} median={st['q50']:.6f} q10={st['q10']:.6f} q90={st['q90']:.6f}")

    # ---------------------------------------------------------
    # Save diagnostics
    # ---------------------------------------------------------
    def topk(triples, k=50, reverse=True):
        return sorted(triples, key=lambda x: x[0], reverse=reverse)[:k]

    scale0_mismatch_samples_top = topk(scale0_mismatch_samples, min(args.max_print, len(scale0_mismatch_samples)))
    scale0_cross_action_samples_top = topk(scale0_cross_action_samples, min(args.max_print, len(scale0_cross_action_samples)))
    dpsnr_bad_top = topk(dpsnr_bad, min(args.max_print, len(dpsnr_bad)))
    dssim_bad_top = topk(dssim_bad, min(args.max_print, len(dssim_bad)))
    huge_delta_top = topk(huge_delta, min(args.max_print, len(huge_delta)))

    print(f"\n[scale0 mismatch samples] top-{len(scale0_mismatch_samples_top)}")
    for t in scale0_mismatch_samples_top:
        print("  ", t)

    print(f"\n[scale0 cross-action samples] top-{len(scale0_cross_action_samples_top)}")
    for t in scale0_cross_action_samples_top:
        print("  ", t)

    print(f"\n[d_psnr recompute mismatch] top-{len(dpsnr_bad_top)}")
    for t in dpsnr_bad_top:
        print("  ", t)

    print(f"\n[d_ssim recompute mismatch] top-{len(dssim_bad_top)}")
    for t in dssim_bad_top:
        print("  ", t)

    print(f"\n[huge ΔPSNR (>|{args.huge_delta}|)] top-{len(huge_delta_top)}")
    for t in huge_delta_top:
        print("  ", t)

    # Save files
    cm_path = os.path.join(args.out_dir, "confusion_matrix.txt")
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(f"baseline_mode={args.baseline_mode}\n")
        f.write(header + "\n")
        for i, ra in enumerate(ACTIONS):
            row = [conf[i][j] for j in range(len(ACTIONS))]
            f.write(f"{ra:>7s} " + " ".join([f"{v:7d}" for v in row]) + "\n")

    summary = {
        "items": len(items),
        "baseline_mode": args.baseline_mode,
        "scale0": args.scale0,
        "abs_psnr0_minus_baseline": stats(abs_psnr0_minus_baseline),
        "abs_ssim0_minus_baseline": stats(abs_ssim0_minus_baseline),
        "range_psnr0_across_actions": stats(range_psnr0_across_actions),
        "range_ssim0_across_actions": stats(range_ssim0_across_actions),
        "dpsnr_recompute_abs_err": stats(dpsnr_recompute_abs_err),
        "dssim_recompute_abs_err": stats(dssim_recompute_abs_err),
        "best_acc": acc,
        "best_correct": best_correct,
        "best_total": best_total,
    }

    with open(os.path.join(args.out_dir, "sanity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_jsonl(os.path.join(args.out_dir, "scale0_mismatch_samples.jsonl"),
               [{"abs_diff": t[0], "idx": t[1], "input": t[2]} for t in scale0_mismatch_samples_top])
    write_jsonl(os.path.join(args.out_dir, "scale0_cross_action_samples.jsonl"),
               [{"range": t[0], "idx": t[1], "input": t[2]} for t in scale0_cross_action_samples_top])
    write_jsonl(os.path.join(args.out_dir, "dpsnr_bad.jsonl"),
               [{"abs_err": t[0], "idx": t[1], "input": t[2], "action": t[3], "scale": t[4]} for t in dpsnr_bad_top])
    write_jsonl(os.path.join(args.out_dir, "dssim_bad.jsonl"),
               [{"abs_err": t[0], "idx": t[1], "input": t[2], "action": t[3], "scale": t[4]} for t in dssim_bad_top])
    write_jsonl(os.path.join(args.out_dir, "huge_delta.jsonl"),
               [{"d_psnr": t[0], "idx": t[1], "input": t[2], "action": t[3], "scale": t[4]} for t in huge_delta_top])

    print(f"\n[Save] confusion matrix -> {cm_path}")
    print(f"[Save] summary -> {os.path.join(args.out_dir, 'sanity_summary.json')}")
    print(f"[Done]")

if __name__ == "__main__":
    main()
