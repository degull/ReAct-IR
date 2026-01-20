# e:/ReAct-IR/scripts/check_toolbank_vs_backbone.py
# ------------------------------------------------------------
# Verify:
#  1) ToolBank ckpt backbone ("backbone.*" keys) is ARCH-COMPATIBLE with a given backbone ckpt ("state_dict")
#  2) ToolBank ckpt backbone weights are SAME / DIFFERENT vs that backbone ckpt (value comparison)
#
# Usage (PowerShell):
#   python e:/ReAct-IR/scripts/check_toolbank_vs_backbone.py `
#     --toolbank_ckpt E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth `
#     --backbone_ckpt E:/ReAct-IR/checkpoints/backbone/epoch_100_L0.1234_P00.00_S0.0000.pth
#
# Optional:
#   --strict_arch 1      # fail if any shape mismatch
#   --show_top 40        # show top-40 largest diffs
#   --sample_equal 20    # sample keys to print equalities
# ------------------------------------------------------------

import os
import sys
import argparse
from typing import Dict, Tuple, Any, List

import torch


# ----------------------------
# Load state_dict from ckpt
# ----------------------------
def load_any_state_dict(path: str) -> Tuple[Dict[str, torch.Tensor], str]:
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        # common containers
        for k in ["toolbank", "state_dict", "model", "net", "network", "backbone", "ema", "params"]:
            v = ckpt.get(k, None)
            if isinstance(v, dict) and len(v) > 0:
                if any(torch.is_tensor(x) for x in v.values()):
                    return v, k

        # raw dict of tensors
        if len(ckpt) > 0 and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, "<root-dict>"

        raise KeyError(f"No usable state_dict in dict-ckpt. keys={list(ckpt.keys())}")

    # directly-saved OrderedDict
    if hasattr(ckpt, "keys") and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, "<root-obj>"

    raise TypeError(f"Unsupported ckpt type: {type(ckpt)}")


def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def strip_known_wrappers(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = sd
    # DataParallel
    if any(k.startswith("module.") for k in out.keys()):
        out = strip_prefix(out, "module.")
    # some saves may include an extra prefix
    if any(k.startswith("toolbank.") for k in out.keys()):
        out = strip_prefix(out, "toolbank.")
    return out


def extract_toolbank_backbone(sd_tool: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Your ToolBank saves: {"toolbank": toolbank.state_dict(), ...}
    toolbank.state_dict() includes:
      - backbone.* (base model)
      - (possibly) adapter/lora params (names vary)
    We extract ONLY backbone.* then strip the prefix.
    """
    bb: Dict[str, torch.Tensor] = {}
    for k, v in sd_tool.items():
        if k.startswith("backbone."):
            bb[k[len("backbone."):]] = v
    return bb


def infer_dim(sd_backbone: Dict[str, torch.Tensor]) -> int:
    # patch_embed.weight: (dim, 3, 3, 3) typical
    w = sd_backbone.get("patch_embed.weight", None)
    if torch.is_tensor(w) and w.ndim == 4:
        return int(w.shape[0])
    # output.weight: (3, dim, 3, 3)
    w = sd_backbone.get("output.weight", None)
    if torch.is_tensor(w) and w.ndim == 4:
        return int(w.shape[1])
    return -1


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = torch.norm(b) + 1e-12
    return float(torch.norm(a - b) / denom)


def tensor_summary(t: torch.Tensor) -> str:
    t = t.detach().cpu().float()
    return f"shape={tuple(t.shape)} dtype={t.dtype} min={t.min().item():.3e} max={t.max().item():.3e} mean={t.mean().item():.3e}"


# ----------------------------
# Main compare
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolbank_ckpt", required=True)
    ap.add_argument("--backbone_ckpt", required=True)

    ap.add_argument("--strict_arch", type=int, default=1, help="1: fail on any shape mismatch (recommended)")
    ap.add_argument("--show_top", type=int, default=30, help="show top-K largest diffs")
    ap.add_argument("--sample_equal", type=int, default=0, help="print N sample keys that are exactly equal")
    args = ap.parse_args()

    tool_path = args.toolbank_ckpt
    bb_path = args.backbone_ckpt

    if not os.path.isfile(tool_path):
        raise FileNotFoundError(f"toolbank_ckpt not found: {tool_path}")
    if not os.path.isfile(bb_path):
        raise FileNotFoundError(f"backbone_ckpt not found: {bb_path}")

    # load
    sd_tool_raw, tool_container = load_any_state_dict(tool_path)
    sd_bb_raw, bb_container = load_any_state_dict(bb_path)

    sd_tool_raw = strip_known_wrappers(sd_tool_raw)
    sd_bb_raw = strip_known_wrappers(sd_bb_raw)

    # extract backbone subdict from toolbank
    sd_tool_bb = extract_toolbank_backbone(sd_tool_raw)

    print("=" * 90)
    print("[TOOLBANK CKPT]")
    print(" path     :", tool_path)
    print(" container:", tool_container)
    print(" keys     :", len(sd_tool_raw))
    print(" bb_keys  :", len(sd_tool_bb))
    print("-" * 90)
    print("[BACKBONE CKPT]")
    print(" path     :", bb_path)
    print(" container:", bb_container)
    print(" keys     :", len(sd_bb_raw))
    print("=" * 90)

    if len(sd_tool_bb) == 0:
        # This means toolbank.state_dict() did not contain "backbone.*"
        # i.e., your ToolBank implementation may save backbone params differently.
        # But given your eval script uses infer_dim_from_toolbank_sd on backbone.patch_embed.weight,
        # it SHOULD contain backbone.*.
        raise RuntimeError(
            "No 'backbone.*' keys found inside ToolBank ckpt state_dict.\n"
            "This suggests ToolBank.state_dict() naming is different. "
            "Search keys containing 'backbone' in your ckpt or adjust extractor."
        )

    # Infer dim (architecture check)
    dim_tool = infer_dim(sd_tool_bb)
    dim_bb = infer_dim(sd_bb_raw)
    print(f"[Infer] dim(toolbank_backbone)={dim_tool} | dim(backbone_ckpt)={dim_bb}")
    if dim_tool != -1 and dim_bb != -1 and dim_tool != dim_bb:
        print("[ARCH][FAIL] dim mismatch => NOT the same backbone config/architecture.")
    else:
        print("[ARCH] dim check: OK (or unknown)")

    # Key matching
    tool_keys = set(sd_tool_bb.keys())
    bb_keys = set(sd_bb_raw.keys())
    common = sorted(list(tool_keys & bb_keys))
    only_tool = sorted(list(tool_keys - bb_keys))
    only_bb = sorted(list(bb_keys - tool_keys))

    print("-" * 90)
    print(f"[KEYS] common={len(common)} | only_toolbank_backbone={len(only_tool)} | only_backbone_ckpt={len(only_bb)}")

    # Shape check on common keys
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for k in common:
        a = sd_tool_bb[k]
        b = sd_bb_raw[k]
        if a.shape != b.shape:
            shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))

    if shape_mismatch:
        print(f"[ARCH][FAIL] shape mismatches on common keys: {len(shape_mismatch)}")
        for k, sa, sb in shape_mismatch[:50]:
            print(f"  - {k}: tool={sa} vs bb={sb}")
        if int(args.strict_arch) == 1:
            raise SystemExit("Strict arch check enabled: exiting due to shape mismatch.")
    else:
        print("[ARCH] shape check: OK (no mismatches on common keys)")

    # If too few common keys, it's effectively incompatible
    if len(common) < 100:
        print("[WARN] Very few common keys. This likely indicates a naming/architecture mismatch.")
        if only_tool:
            print("[Only toolbank backbone keys] sample:")
            for k in only_tool[:30]:
                print(" ", k)
        if only_bb:
            print("[Only backbone ckpt keys] sample:")
            for k in only_bb[:30]:
                print(" ", k)

    # Value comparisons (same backbone weights?)
    diffs = []
    exact_equal = 0
    compared = 0

    for k in common:
        a = sd_tool_bb[k]
        b = sd_bb_raw[k]
        if a.shape != b.shape:
            continue
        compared += 1
        if torch.equal(a, b):
            exact_equal += 1
            continue
        diffs.append((k, rel_l2(a, b), max_abs(a, b), tuple(a.shape)))

    ratio = exact_equal / max(1, compared)
    print("-" * 90)
    print(f"[WEIGHT] compared={compared} exact_equal={exact_equal} exact_equal_ratio={ratio:.4f}")

    # Decision guidance
    # - If toolbank trained with frozen backbone and started from that backbone ckpt:
    #   exact_equal_ratio should be extremely high (often ~1.0)
    # - If backbone was different init or backbone kept training elsewhere:
    #   ratio will drop, but still architecture may match.
    if compared == 0:
        print("[WEIGHT][FAIL] No comparable keys (after filtering shape mismatches).")
    else:
        if ratio > 0.99:
            print("[DECISION] SAME backbone weights (ToolBank ckpt backbone == this backbone ckpt).")
        elif ratio > 0.90:
            print("[DECISION] VERY similar backbone weights (likely same run but not identical save moment).")
        elif ratio > 0.50:
            print("[DECISION] PARTIALLY similar (architecture matches but weights differ significantly).")
        else:
            print("[DECISION] DIFFERENT backbone weights (ToolBank NOT trained on this backbone weights).")

    # Show top diffs
    if diffs:
        diffs.sort(key=lambda x: (x[2], x[1]), reverse=True)  # by max_abs then rel_l2
        kshow = min(int(args.show_top), len(diffs))
        print("-" * 90)
        print(f"[TOP DIFFS] showing {kshow}/{len(diffs)} by max_abs")
        for k, r, m, sh in diffs[:kshow]:
            print(f"  {k:60s}  max_abs={m:.3e}  rel_l2={r:.3e}  shape={sh}")

        # show a couple tensor summaries for the worst key
        k0 = diffs[0][0]
        print("-" * 90)
        print("[WORST KEY DETAILS]")
        print(" key:", k0)
        print(" tool:", tensor_summary(sd_tool_bb[k0]))
        print(" bb  :", tensor_summary(sd_bb_raw[k0]))

    # Optionally print some equal keys
    if int(args.sample_equal) > 0 and exact_equal > 0:
        import random
        eq_keys = [k for k in common if (sd_tool_bb[k].shape == sd_bb_raw[k].shape and torch.equal(sd_tool_bb[k], sd_bb_raw[k]))]
        random.shuffle(eq_keys)
        n = min(int(args.sample_equal), len(eq_keys))
        print("-" * 90)
        print(f"[EQUAL SAMPLE] showing {n}/{len(eq_keys)}")
        for k in eq_keys[:n]:
            print("  ", k)

    # Print missing key samples (for debugging naming changes)
    if only_tool:
        print("-" * 90)
        print("[ONLY TOOLBANK BACKBONE KEYS] first 40")
        for k in only_tool[:40]:
            print("  ", k)

    if only_bb:
        print("-" * 90)
        print("[ONLY BACKBONE CKPT KEYS] first 40")
        for k in only_bb[:40]:
            print("  ", k)

    print("=" * 90)
    print("[DONE]")
    print("Interpretation:")
    print(" - ARCH compatible: dim OK + no shape mismatches on common keys + large common key count")
    print(" - SAME backbone weights: exact_equal_ratio ~ 1.0")
    print(" - If arch OK but weights differ: ToolBank ckpt was trained on a different backbone init/weights")
    print("=" * 90)


if __name__ == "__main__":
    main()
# ToolBank ckpt vs Backbone ckpt 비교 실행 예시 (PowerShell)

