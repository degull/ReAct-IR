# LoRA 적용 결과(이미지별 PSNR/SSIM)를 모아서 라벨 JSON 생성
# scripts/make_diagnoser_labels.py
# E:\ReAct-IR\scripts\make_diagnoser_labels.py
# ------------------------------------------------------------
# Build Diagnoser labels from per-image eval jsonl files:
#   - input:  metrics_scale_0.jsonl (scale=0)
#   - input:  metrics_scale_1.jsonl (scale=1)
#   - output: diagnoser_labels.json  (merged across actions)
#
# Label definition (recommended):
#   ΔQ = ΔPSNR + lam * (100 * ΔSSIM)
#   score = clip( ΔQ / qmax , 0, 1 )
#
# Each jsonl line expected:
#   {"id": "...", "inp": "...", "gt": "...", "psnr": 29.3, "ssim": 0.95, "action": "...", "scale": 0.0}
#
# ------------------------------------------------------------
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def _as_posix(p: str) -> str:
    return str(p).replace("\\", "/")


def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: id -> record
    If duplicate ids exist, keeps the last one.
    """
    data: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSON parse failed at {path}:{ln}: {e}")
            if "id" not in obj:
                raise RuntimeError(f"Missing 'id' field at {path}:{ln}")
            _id = str(obj["id"])
            data[_id] = obj
    return data


def compute_dq(
    psnr0: float,
    ssim0: float,
    psnr1: float,
    ssim1: float,
    lam: float,
) -> Tuple[float, float, float]:
    dpsnr = float(psnr1 - psnr0)
    dssim = float(ssim1 - ssim0)
    dQ = float(dpsnr + lam * (100.0 * dssim))
    return dpsnr, dssim, dQ


def normalize_score(
    dQ: float,
    qmax: float,
    clamp_min: float,
    clamp_max: Optional[float],
) -> float:
    # clamp dQ first (recommended to prevent crazy outliers)
    dQ2 = float(dQ)
    if clamp_max is not None:
        dQ2 = max(float(clamp_min), min(float(clamp_max), dQ2))
    else:
        dQ2 = max(float(clamp_min), dQ2)

    denom = max(1e-8, float(qmax))
    s = dQ2 / denom
    # force [0,1]
    s = max(0.0, min(1.0, float(s)))
    return s


def load_or_init_labels(out_path: Path) -> Dict[str, Any]:
    if out_path.exists():
        try:
            obj = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("root must be a dict")
            obj.setdefault("meta", {})
            obj.setdefault("items", {})
            if not isinstance(obj["meta"], dict) or not isinstance(obj["items"], dict):
                raise ValueError("meta/items must be dict")
            return obj
        except Exception as e:
            raise RuntimeError(f"Failed to read existing labels json: {out_path} ({e})")
    return {"meta": {}, "items": {}}


def infer_action_name(
    *,
    action_arg: Optional[str],
    scale0_path: Path,
    scale1_path: Path,
) -> str:
    """
    If --action is provided, use it.
    Else try to infer from the jsonl records or from directory name.
    """
    if action_arg and str(action_arg).strip():
        return str(action_arg).strip().lower()

    # try read first non-empty line from scale0 and look at obj["action"]
    for p in [scale0_path, scale1_path]:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "action" in obj and str(obj["action"]).strip():
                        return str(obj["action"]).strip().lower()
                except Exception:
                    break

    # fallback to parent directory name (e.g., .../dehaze/...)
    parts = [x.lower() for x in scale0_path.parts]
    # find common aliases
    for alias in ["dedrop", "desnow", "derain", "deblur", "dehaze"]:
        if alias in parts:
            return alias
    return "unknown_action"


def validate_pairing(
    scale0: Dict[str, Dict[str, Any]],
    scale1: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, int]:
    """
    Returns (n0, n1, n_common)
    """
    n0 = len(scale0)
    n1 = len(scale1)
    common = len(set(scale0.keys()) & set(scale1.keys()))
    return n0, n1, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="", help="action alias, e.g. dehaze, derain ... (optional; can be inferred)")
    ap.add_argument("--scale0", required=True, help="jsonl per-image metrics for scale=0")
    ap.add_argument("--scale1", required=True, help="jsonl per-image metrics for scale=1")
    ap.add_argument("--out", required=True, help="output labels json (created/updated)")

    # ΔQ = ΔPSNR + lam*(100*ΔSSIM)
    ap.add_argument("--lam", type=float, default=0.2, help="lambda for SSIM term (default=0.2)")
    # score = clip(ΔQ/qmax, 0, 1)
    ap.add_argument("--qmax", type=float, default=6.0, help="normalization upper bound for ΔQ (default=6.0)")

    # optional dQ clamp before normalization
    ap.add_argument("--clamp_min", type=float, default=0.0, help="min clamp for ΔQ before normalization (default=0)")
    ap.add_argument("--clamp_max", type=float, default=None, help="max clamp for ΔQ before normalization (default=None)")

    # bookkeeping
    ap.add_argument("--overwrite_action", type=int, default=0, help="if 1: overwrite existing action label for same ids")

    args = ap.parse_args()

    scale0_path = Path(args.scale0)
    scale1_path = Path(args.scale1)
    out_path = Path(args.out)

    if not scale0_path.is_file():
        raise FileNotFoundError(f"--scale0 not found: {scale0_path}")
    if not scale1_path.is_file():
        raise FileNotFoundError(f"--scale1 not found: {scale1_path}")

    action = infer_action_name(action_arg=args.action, scale0_path=scale0_path, scale1_path=scale1_path)
    if action == "unknown_action":
        print("[Warn] action could not be inferred; using 'unknown_action'. Consider passing --action explicitly.")

    scale0 = read_jsonl(scale0_path)
    scale1 = read_jsonl(scale1_path)

    n0, n1, n_common = validate_pairing(scale0, scale1)
    if n_common == 0:
        raise RuntimeError(
            "No common ids between scale0 and scale1 jsonl files.\n"
            f"  scale0: {scale0_path} (n={n0})\n"
            f"  scale1: {scale1_path} (n={n1})\n"
            "Make sure 'id' is identical across scales (your eval script uses meta['key'])."
        )

    labels = load_or_init_labels(out_path)
    meta = labels.setdefault("meta", {})
    items = labels.setdefault("items", {})

    # update meta
    meta.setdefault("actions", [])
    if action not in meta["actions"]:
        meta["actions"].append(action)
    meta["actions"] = sorted(list(set(meta["actions"])))

    meta["dq_formula"] = "dQ = dPSNR + lam*(100*dSSIM)"
    meta["score_formula"] = "score = clip(dQ/qmax, 0, 1)"
    meta["lam"] = float(args.lam)
    meta["qmax"] = float(args.qmax)
    meta["clamp_min"] = float(args.clamp_min)
    meta["clamp_max"] = (None if args.clamp_max is None else float(args.clamp_max))
    meta.setdefault("sources", {})
    meta["sources"][action] = {
        "scale0": _as_posix(str(scale0_path)),
        "scale1": _as_posix(str(scale1_path)),
        "n_scale0": int(n0),
        "n_scale1": int(n1),
        "n_common": int(n_common),
    }

    # build per-item labels
    common_ids = sorted(set(scale0.keys()) & set(scale1.keys()))
    added, skipped, updated = 0, 0, 0

    # stats
    dQ_min, dQ_max, dQ_sum = 1e9, -1e9, 0.0
    sc_min, sc_max, sc_sum = 1e9, -1e9, 0.0

    for _id in common_ids:
        r0 = scale0[_id]
        r1 = scale1[_id]

        # required fields
        for k in ["psnr", "ssim"]:
            if k not in r0 or k not in r1:
                raise RuntimeError(f"Missing '{k}' for id={_id} in scale0 or scale1 jsonl.")

        psnr0 = float(r0["psnr"])
        ssim0 = float(r0["ssim"])
        psnr1 = float(r1["psnr"])
        ssim1 = float(r1["ssim"])

        dpsnr, dssim, dQ = compute_dq(psnr0, ssim0, psnr1, ssim1, lam=float(args.lam))
        score = normalize_score(dQ, qmax=float(args.qmax), clamp_min=float(args.clamp_min), clamp_max=args.clamp_max)

        dQ_min = min(dQ_min, dQ)
        dQ_max = max(dQ_max, dQ)
        dQ_sum += dQ

        sc_min = min(sc_min, score)
        sc_max = max(sc_max, score)
        sc_sum += score

        # merge into output
        it = items.get(_id, {})
        if not isinstance(it, dict):
            it = {}

        # keep representative paths
        inp_path = r0.get("inp", it.get("inp", ""))
        gt_path = r0.get("gt", it.get("gt", ""))
        it["inp"] = _as_posix(str(inp_path)) if inp_path else ""
        it["gt"] = _as_posix(str(gt_path)) if gt_path else ""

        it.setdefault("label_raw", {})
        it.setdefault("label_score", {})

        if (not args.overwrite_action) and (action in it["label_score"]):
            skipped += 1
            items[_id] = it
            continue

        it["label_raw"][action] = {
            "psnr0": psnr0,
            "ssim0": ssim0,
            "psnr1": psnr1,
            "ssim1": ssim1,
            "dpsnr": dpsnr,
            "dssim": dssim,
            "dQ": dQ,
        }
        it["label_score"][action] = float(score)

        if _id not in items:
            added += 1
        else:
            updated += 1
        items[_id] = it

    # summary meta stats
    n_used = max(1, (added + updated + skipped))
    meta.setdefault("stats", {})
    meta["stats"][action] = {
        "added": int(added),
        "updated": int(updated),
        "skipped_existing": int(skipped),
        "dQ_mean": float(dQ_sum / max(1, (added + updated))),
        "dQ_min": float(dQ_min if dQ_min < 1e8 else 0.0),
        "dQ_max": float(dQ_max if dQ_max > -1e8 else 0.0),
        "score_mean": float(sc_sum / max(1, (added + updated))),
        "score_min": float(sc_min if sc_min < 1e8 else 0.0),
        "score_max": float(sc_max if sc_max > -1e8 else 0.0),
        "n_common": int(n_common),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote labels:", out_path)
    print(f"  action={action}")
    print(f"  scale0={scale0_path} (n={n0})")
    print(f"  scale1={scale1_path} (n={n1})")
    print(f"  common_ids={n_common}")
    print(f"  added={added} updated={updated} skipped_existing={skipped}")
    print(f"  dQ: mean={meta['stats'][action]['dQ_mean']:.4f} min={meta['stats'][action]['dQ_min']:.4f} max={meta['stats'][action]['dQ_max']:.4f}")
    print(f"  score: mean={meta['stats'][action]['score_mean']:.4f} min={meta['stats'][action]['score_min']:.4f} max={meta['stats'][action]['score_max']:.4f}")


if __name__ == "__main__":
    main()


"""
# dehaze
python E:\ReAct-IR\scripts\make_diagnoser_labels.py ^
  --action dehaze ^
  --scale0 "E:/ReAct-IR/results/lora_eval/dehaze/test_all/scale_0/metrics_scale_0.jsonl" ^
  --scale1 "E:/ReAct-IR/results/lora_eval/dehaze/test_all/scale_1/metrics_scale_1.jsonl" ^
  --out    "E:/ReAct-IR/results/diagnoser_labels.json" ^
  --lam 0.2 --qmax 6.0

  
# derain

# desnow

# deblur

# dedrop

"""