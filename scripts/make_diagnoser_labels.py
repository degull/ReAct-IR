# LoRA 적용 결과(이미지별 PSNR/SSIM)를 모아서 라벨 JSON 생성# scripts/make_diagnoser_labels.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any

def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = str(obj["id"])
            data[_id] = obj
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action", required=True, help="e.g., dehaze, desnow, deblur ... (label key)")
    p.add_argument("--scale0", required=True, help="jsonl per-image metrics at scale=0")
    p.add_argument("--scale1", required=True, help="jsonl per-image metrics at scale=1")
    p.add_argument("--out", required=True, help="output labels json (will be created/updated)")

    # ΔQ = ΔPSNR + lam * (100 * ΔSSIM)
    p.add_argument("--lam", type=float, default=0.2, help="lambda for SSIM term (default=0.2)")
    p.add_argument("--qmax", type=float, default=6.0, help="normalization upper bound for ΔQ (default=6.0)")
    p.add_argument("--clamp_min", type=float, default=0.0)
    p.add_argument("--clamp_max", type=float, default=None)

    args = p.parse_args()

    action = args.action.lower()
    s0 = read_jsonl(Path(args.scale0))
    s1 = read_jsonl(Path(args.scale1))

    common_ids = sorted(set(s0.keys()) & set(s1.keys()))
    if not common_ids:
        raise RuntimeError("No common ids between scale0 and scale1 jsonl files.")

    out_path = Path(args.out)
    if out_path.exists():
        labels = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        labels = {"meta": {}, "items": {}}

    labels["meta"].setdefault("actions", [])
    if action not in labels["meta"]["actions"]:
        labels["meta"]["actions"].append(action)

    labels["meta"]["lam"] = args.lam
    labels["meta"]["qmax"] = args.qmax
    labels["meta"]["note"] = "score = clip(ΔQ/qmax, 0, 1), where ΔQ = ΔPSNR + lam*(100*ΔSSIM), using scale=1 vs scale=0"

    clamp_min = args.clamp_min
    clamp_max = args.clamp_max

    added = 0
    for _id in common_ids:
        psnr0, ssim0 = float(s0[_id]["psnr"]), float(s0[_id]["ssim"])
        psnr1, ssim1 = float(s1[_id]["psnr"]), float(s1[_id]["ssim"])
        dpsnr = psnr1 - psnr0
        dssim = ssim1 - ssim0

        dQ = dpsnr + args.lam * (100.0 * dssim)

        if clamp_max is not None:
            dQ = max(clamp_min, min(clamp_max, dQ))
        else:
            dQ = max(clamp_min, dQ)

        score = dQ / max(1e-8, args.qmax)
        score = max(0.0, min(1.0, score))

        item = labels["items"].get(_id, {})
        # inp/gt는 scale0 파일의 것을 대표로 사용
        item["inp"] = s0[_id].get("inp", item.get("inp"))
        item["gt"] = s0[_id].get("gt", item.get("gt"))

        item.setdefault("label_raw", {})
        item.setdefault("label_score", {})

        item["label_raw"][action] = {
            "psnr0": psnr0, "ssim0": ssim0,
            "psnr1": psnr1, "ssim1": ssim1,
            "dpsnr": dpsnr, "dssim": dssim,
            "dQ": dQ
        }
        item["label_score"][action] = float(score)

        labels["items"][_id] = item
        added += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] action={action} wrote/updated {added} items into: {out_path}")

if __name__ == "__main__":
    main()
