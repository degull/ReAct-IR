# scripts/rebase_rollouts.py
import os
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

# -------------------------------------------------------
# IO utils
# -------------------------------------------------------
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

# -------------------------------------------------------
# Core
# -------------------------------------------------------
def find_scale0_entry(sweep: List[Dict[str, Any]], scale0: float, prefer_action: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Find a sweep entry with scale==scale0.
    If prefer_action is provided, prefer that action's scale0 entry.
    Otherwise return the first match.
    """
    if not sweep:
        return None

    # Prefer a specific action if asked
    if prefer_action is not None:
        for e in sweep:
            if e.get("action") == prefer_action and float(e.get("scale", -999)) == float(scale0):
                return e

    # Any action's scale0 (should be identical across actions by design)
    for e in sweep:
        if float(e.get("scale", -999)) == float(scale0):
            return e

    return None

def recompute_best_by_dpsnr(sweep: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Choose best by (d_psnr, then d_ssim) among sweep entries.
    """
    best = None
    best_key = None
    for e in sweep:
        dps = e.get("d_psnr", None)
        dss = e.get("d_ssim", None)
        if dps is None or dss is None or (not is_finite(dps)) or (not is_finite(dss)):
            continue
        key = (float(dps), float(dss))
        if best is None or key > best_key:
            best = e
            best_key = key
    if best is None:
        return None
    # return a compact dict
    return {
        "action": best.get("action"),
        "scale": float(best.get("scale")),
        "psnr": float(best.get("psnr")),
        "ssim": float(best.get("ssim")),
        "d_psnr": float(best.get("d_psnr")),
        "d_ssim": float(best.get("d_ssim")),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, type=str)
    ap.add_argument("--out_jsonl", required=True, type=str)

    ap.add_argument("--scale0", default=0.0, type=float, help="scale value treated as backbone baseline")
    ap.add_argument("--tol", default=1e-6, type=float, help="tolerance for baseline checks")

    ap.add_argument("--prefer_action", default="", type=str,
                    help="optional: prefer this action's scale0 entry as baseline (e.g., A_DEBLUR)")
    ap.add_argument("--rewrite_best", default=1, type=int, help="1: recompute best based on new deltas")
    ap.add_argument("--keep_old_best", default=1, type=int, help="1: store previous best as best_old")

    args = ap.parse_args()

    prefer_action = args.prefer_action.strip() if args.prefer_action.strip() else None

    print(f"[Load] {args.in_jsonl}")
    items = read_jsonl(args.in_jsonl)
    print(f"[Load] items = {len(items)}")

    ok = 0
    skip = 0
    skip_missing_sweep = 0
    skip_missing_scale0 = 0

    out_items = []

    for it in items:
        sweep = it.get("sweep", None)
        if not isinstance(sweep, list) or len(sweep) == 0:
            skip += 1
            skip_missing_sweep += 1
            continue

        scale0_entry = find_scale0_entry(sweep, args.scale0, prefer_action)
        if scale0_entry is None:
            skip += 1
            skip_missing_scale0 += 1
            continue

        # baseline becomes scale0 outputs
        psnr0 = scale0_entry.get("psnr", None)
        ssim0 = scale0_entry.get("ssim", None)
        if psnr0 is None or ssim0 is None or (not is_finite(psnr0)) or (not is_finite(ssim0)):
            skip += 1
            skip_missing_scale0 += 1
            continue
        psnr0 = float(psnr0)
        ssim0 = float(ssim0)

        # Keep a copy of old baseline fields, but add new baseline keys
        baseline = it.get("baseline", {})
        if not isinstance(baseline, dict):
            baseline = {}

        # Preserve original input baseline if present
        # and set new "base" baseline
        baseline["psnr_base"] = psnr0
        baseline["ssim_base"] = ssim0
        baseline["base_scale"] = float(args.scale0)
        baseline["base_from_action"] = scale0_entry.get("action", None)

        it["baseline"] = baseline

        # Recompute deltas for all sweep entries w.r.t base
        for e in sweep:
            ps = e.get("psnr", None)
            ss = e.get("ssim", None)
            if ps is None or ss is None or (not is_finite(ps)) or (not is_finite(ss)):
                e["d_psnr"] = None
                e["d_ssim"] = None
                continue
            ps = float(ps)
            ss = float(ss)
            e["d_psnr"] = ps - psnr0
            e["d_ssim"] = ss - ssim0

        it["sweep"] = sweep

        # best rewrite
        if args.keep_old_best == 1 and "best" in it:
            it["best_old"] = it["best"]

        if args.rewrite_best == 1:
            new_best = recompute_best_by_dpsnr(sweep)
            if new_best is not None:
                it["best"] = new_best

        out_items.append(it)
        ok += 1

    write_jsonl(args.out_jsonl, out_items)

    print("\n==================== Rebase Report (scale0 baseline) ====================")
    print(f"[Input ] {args.in_jsonl}")
    print(f"[Output] {args.out_jsonl}")
    print(f"[Items ] in={len(items)} out={len(out_items)}")
    print(f"[OK    ] {ok}")
    print(f"[Skip  ] {skip}")
    print(f"[Skip 이유] missing_sweep={skip_missing_sweep} missing_scale0={skip_missing_scale0}")
    print(f"[Baseline] scale0={args.scale0} tol={args.tol} prefer_action={prefer_action}")
    print("[Done]")

if __name__ == "__main__":
    main()
