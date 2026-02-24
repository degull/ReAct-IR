# tools/compare_lora_actions.py
# ------------------------------------------------------------
# Compare action-wise LoRA checkpoints to verify they are truly different.
#
# What it does:
#  (A) Weight-level comparison (always works):
#      - loads LoRA-only checkpoints (expects ckpt["lora_state_dict"] or plain state_dict)
#      - computes per-layer stats: L2 norm, cosine similarity, L2 distance between actions
#      - writes CSV reports + prints top-k differing layers
#
#  (B) Effect-level comparison (optional):
#      - if project modules are importable (VETNet, ToolBank), run:
#         y_base = backbone(x) with A_STOP (or no-op)
#         y_act  = backbone(x) with action LoRA active
#         delta  = |y_act - y_base|
#      - saves delta maps as PNG
#
# Usage examples are at bottom.
# ------------------------------------------------------------

import os
import sys
import glob
import math
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


# -----------------------------
# Helpers: IO
# -----------------------------
def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def find_latest_ckpt(action_dir: str) -> str:
    """
    Finds latest epoch_XXX_*.pth by epoch number.
    """
    files = glob.glob(os.path.join(action_dir, "epoch_*.pth"))
    if not files:
        raise FileNotFoundError(f"No epoch_*.pth found in: {action_dir}")
    best = None
    best_epoch = -1
    for f in files:
        base = os.path.basename(f)
        # epoch_001_....pth
        try:
            ep = int(base.split("_")[1])
        except Exception:
            continue
        if ep > best_epoch:
            best_epoch = ep
            best = f
    if best is None:
        # fallback
        files.sort()
        best = files[-1]
    return best


def load_lora_sd(ckpt_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Supports:
      - ckpt dict with "lora_state_dict"
      - plain state_dict
    Returns: dict(str -> tensor on CPU)
    """
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "lora_state_dict" in obj and isinstance(obj["lora_state_dict"], dict):
        sd = obj["lora_state_dict"]
    elif isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        sd = obj
    else:
        raise RuntimeError(f"Unsupported ckpt format: {ckpt_path}")

    # ensure cpu tensors
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
    if not out:
        raise RuntimeError(f"No tensors found in LoRA state_dict: {ckpt_path}")
    return out


def list_actions_from_root(lora_root: str) -> List[str]:
    actions = []
    for name in os.listdir(lora_root):
        p = os.path.join(lora_root, name)
        if os.path.isdir(p):
            actions.append(name)
    actions.sort()
    return actions


# -----------------------------
# Helpers: vector stats
# -----------------------------
def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(-1).float()


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = _flatten(a)
    b = _flatten(b)
    na = float(torch.norm(a).item())
    nb = float(torch.norm(b).item())
    if na < eps or nb < eps:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb + eps))


def l2_norm(a: torch.Tensor) -> float:
    return float(torch.norm(_flatten(a)).item())


def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(_flatten(a) - _flatten(b)).item())


def group_base_layer_key(k: str) -> str:
    """
    Example key:
      backbone.encoder1.body.0.attn.qkv.lora_A.A_DEHAZE.weight
    Base key (action-agnostic):
      backbone.encoder1.body.0.attn.qkv.lora_A
    We drop ".A_XXXX" and anything after it.
    """
    parts = k.split(".")
    # find token starting with "A_" (action)
    out = []
    for t in parts:
        if t.startswith("A_"):
            break
        out.append(t)
    # also drop trailing "weight"/"bias" if present (keep module-ish key)
    if out and out[-1] in ("weight", "bias"):
        out = out[:-1]
    return ".".join(out)


def is_lora_tensor_key(k: str) -> bool:
    return (".lora_A." in k) or (".lora_B." in k)


# -----------------------------
# Core: compare weight tensors across actions
# -----------------------------
def compare_actions_weight_level(
    action_to_sd: Dict[str, Dict[str, torch.Tensor]],
    out_dir: str,
    topk: int = 30,
):
    safe_makedirs(out_dir)

    actions = sorted(action_to_sd.keys())
    if len(actions) < 2:
        raise ValueError("Need at least 2 actions to compare.")

    # Build a common key set (only LoRA keys)
    keysets = []
    for a in actions:
        ks = {k for k in action_to_sd[a].keys() if is_lora_tensor_key(k)}
        keysets.append(ks)
    common_keys = set.intersection(*keysets)
    if not common_keys:
        raise RuntimeError("No common LoRA keys found across selected actions (check checkpoints).")

    # Group by base layer key
    base_groups: Dict[str, List[str]] = {}
    for k in sorted(common_keys):
        bk = group_base_layer_key(k)
        base_groups.setdefault(bk, []).append(k)

    # Pairwise comparisons
    pair_rows = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            a1, a2 = actions[i], actions[j]
            # aggregate diffs per base group
            group_scores = []
            for bk, keys in base_groups.items():
                # For each bk, we may have multiple tensors (often just one "weight")
                cos_list = []
                dist_list = []
                n1_list = []
                n2_list = []
                for k in keys:
                    t1 = action_to_sd[a1][k]
                    t2 = action_to_sd[a2][k]
                    cos_list.append(cosine_sim(t1, t2))
                    dist_list.append(l2_dist(t1, t2))
                    n1_list.append(l2_norm(t1))
                    n2_list.append(l2_norm(t2))

                # Reduce
                cos_mean = float(np.mean(cos_list)) if cos_list else 0.0
                dist_sum = float(np.sum(dist_list)) if dist_list else 0.0
                n1_sum = float(np.sum(n1_list)) if n1_list else 0.0
                n2_sum = float(np.sum(n2_list)) if n2_list else 0.0

                group_scores.append((bk, cos_mean, dist_sum, n1_sum, n2_sum))

            # sort by distance descending (most different layers)
            group_scores.sort(key=lambda x: x[2], reverse=True)

            # write per-pair CSV
            csv_path = os.path.join(out_dir, f"pair_{a1}_vs_{a2}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("base_layer,cosine_mean,l2dist_sum,norm_sum_action1,norm_sum_action2\n")
                for bk, cos_m, dist_s, n1_s, n2_s in group_scores:
                    f.write(f"{bk},{cos_m:.6f},{dist_s:.6f},{n1_s:.6f},{n2_s:.6f}\n")

            # keep summary rows
            # overall cosine over all tensors
            all_cos = []
            all_dist = []
            for k in sorted(common_keys):
                all_cos.append(cosine_sim(action_to_sd[a1][k], action_to_sd[a2][k]))
                all_dist.append(l2_dist(action_to_sd[a1][k], action_to_sd[a2][k]))

            pair_rows.append(
                {
                    "pair": f"{a1} vs {a2}",
                    "cosine_mean_all": float(np.mean(all_cos)),
                    "l2dist_mean_all": float(np.mean(all_dist)),
                    "l2dist_sum_all": float(np.sum(all_dist)),
                    "csv": csv_path,
                    "top_groups": group_scores[:topk],
                }
            )

    # write summary CSV
    sum_path = os.path.join(out_dir, "summary_pairs.csv")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("pair,cosine_mean_all,l2dist_mean_all,l2dist_sum_all\n")
        for r in pair_rows:
            f.write(f"{r['pair']},{r['cosine_mean_all']:.6f},{r['l2dist_mean_all']:.6f},{r['l2dist_sum_all']:.6f}\n")

    # print a readable report
    print("\n==================== [LoRA Weight Comparison] ====================")
    print(f"[Actions] {actions}")
    print(f"[Common LoRA keys] {len(common_keys)} tensors")
    print(f"[Output] {out_dir}")
    print(f"[Summary CSV] {sum_path}")

    for r in pair_rows:
        print(f"\n--- {r['pair']} ---")
        print(f"  cosine_mean_all = {r['cosine_mean_all']:.4f}   (1.0 = identical, 0 = unrelated)")
        print(f"  l2dist_sum_all  = {r['l2dist_sum_all']:.4f}")
        print(f"  per-layer CSV   = {r['csv']}")
        print(f"  Top-{topk} most different base layers (by l2dist_sum):")
        for bk, cos_m, dist_s, n1_s, n2_s in r["top_groups"]:
            print(f"    dist={dist_s:10.4f}  cos={cos_m:+.4f}  |norm|=({n1_s:.3f},{n2_s:.3f})  :: {bk}")

    print("\nInterpretation tips:")
    print("- If cosine_mean_all is far below 1.0 and l2dist is large -> they are truly different.")
    print("- If cosine_mean_all ~ 1.0 and l2dist tiny -> they may be nearly identical (bad sign).")


# -----------------------------
# Optional: delta-map (effect-level) comparison
# -----------------------------
def load_image_as_tensor(path: str, patch: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    # simple center-crop/resize to patch
    if img.size[0] != patch or img.size[1] != patch:
        img = img.resize((patch, patch), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    chw = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return chw


def save_delta_map(delta: torch.Tensor, out_path: str):
    """
    delta: 1x3xHxW float tensor (abs difference).
    Save as grayscale heatmap = mean over channels, normalized 0..255.
    """
    d = delta.detach().cpu()
    d = d.mean(dim=1, keepdim=False)[0]  # HxW
    d_np = d.numpy()
    mx = float(d_np.max()) if d_np.size > 0 else 1.0
    if mx < 1e-12:
        mx = 1.0
    d_np = (d_np / mx * 255.0).clip(0, 255).astype(np.uint8)
    safe_makedirs(os.path.dirname(out_path))
    Image.fromarray(d_np, mode="L").save(out_path)


def try_import_project_modules(project_root: str):
    """
    Try to import VETNet/ToolBank from your project.
    If not available, return (None, None).
    """
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from models.backbone.vetnet import VETNet  # type: ignore
        from models.toolbank.toolbank import ToolBank  # type: ignore

        return VETNet, ToolBank
    except Exception as e:
        print("\n[WARN] Could not import project modules for delta-map mode.")
        print("       Weight-level comparison will still work.")
        print(f"       Import error: {repr(e)}")
        return None, None


@torch.no_grad()
def run_delta_maps(
    project_root: str,
    backbone_ckpt: str,
    action_to_lora_ckpt: Dict[str, str],
    out_dir: str,
    dim: int,
    bias: int,
    volterra_rank: int,
    patch: int,
    image_path: str,
    use_amp: int,
    channels_last: int,
):
    VETNet, ToolBank = try_import_project_modules(project_root)
    if VETNet is None or ToolBank is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================== [Delta-Map Mode] ====================")
    print(f"[Device] {device}")
    print(f"[Image] {image_path}")

    x = load_image_as_tensor(image_path, patch=patch).to(device)
    if channels_last and device.type == "cuda":
        x = x.to(memory_format=torch.channels_last)

    # Build & load backbone
    base = VETNet(dim=dim, bias=bool(bias), volterra_rank=volterra_rank)
    ckpt_obj = torch.load(backbone_ckpt, map_location="cpu")
    sd = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    base.load_state_dict(sd, strict=False)

    for p in base.parameters():
        p.requires_grad = False

    tb = ToolBank(
        base,
        actions=["A_DEDROP", "A_DEBLUR", "A_DESNOW", "A_DERAIN", "A_DEHAZE", "A_STOP"],
        rank=2,      # doesn't matter for loading; structure should match training
        alpha=1.0,
        wrap_only_1x1=True,
    ).to(device)

    if channels_last and device.type == "cuda":
        tb = tb.to(memory_format=torch.channels_last)

    tb.eval()

    # baseline output (no-op)
    # safest: activate A_STOP if implemented as no-op scale 0
    # if activate exists:
    if hasattr(tb, "activate"):
        tb.activate("A_STOP", scale=0.0)

    autocast_enabled = bool(use_amp) and (device.type == "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
        y_base = tb(x).clamp(0, 1)

    # load & run each action
    for action_alias, lora_ckpt_path in action_to_lora_ckpt.items():
        obj = torch.load(lora_ckpt_path, map_location="cpu")
        lora_sd = obj["lora_state_dict"] if isinstance(obj, dict) and "lora_state_dict" in obj else obj

        # load action into ToolBank
        if hasattr(tb, "load_lora_state_dict_for_action"):
            # map alias to internal
            internal = {
                "dedrop": "A_DEDROP",
                "deblur": "A_DEBLUR",
                "desnow": "A_DESNOW",
                "derain": "A_DERAIN",
                "dehaze": "A_DEHAZE",
            }[action_alias]
            tb.load_lora_state_dict_for_action(internal, lora_sd, strict=True)
            tb.activate(internal, scale=1.0)
        elif hasattr(tb, "load_lora_state_dict"):
            internal = {
                "dedrop": "A_DEDROP",
                "deblur": "A_DEBLUR",
                "desnow": "A_DESNOW",
                "derain": "A_DERAIN",
                "dehaze": "A_DEHAZE",
            }[action_alias]
            tb.load_lora_state_dict(internal, lora_sd, strict=True)
            tb.activate(internal, scale=1.0)
        else:
            # fallback: try strict=False on whole module
            tb.load_state_dict(lora_sd, strict=False)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            y_act = tb(x).clamp(0, 1)

        delta = (y_act - y_base).abs()
        out_path = os.path.join(out_dir, "delta_maps", f"delta_{action_alias}.png")
        save_delta_map(delta, out_path)
        print(f"[Saved Δmap] {out_path}")

    print("[Delta-Map] done.")


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_root", default=None, help="E:/ReAct-IR (needed only for delta-map mode)")
    ap.add_argument("--lora_root", required=True, help="E:/ReAct-IR/checkpoints/toolbank_lora")
    ap.add_argument(
        "--actions",
        default="dehaze,deblur,derain,dedrop,desnow",
        help="comma-separated action aliases among: dedrop,deblur,desnow,derain,dehaze",
    )
    ap.add_argument("--out_dir", default="E:/ReAct-IR/results/lora_compare", help="output directory")
    ap.add_argument("--topk", type=int, default=30, help="top-K differing layers to print")

    # choose ckpt per action
    ap.add_argument("--ckpt", default="latest", help="'latest' or explicit .pth path (applies to all actions if path)")
    ap.add_argument("--epoch", type=int, default=-1, help="if >0, pick epoch_XXX_*.pth for each action")

    # delta-map mode (optional)
    ap.add_argument("--do_delta", type=int, default=0, help="1 to run delta-map mode (requires project imports)")
    ap.add_argument("--backbone_ckpt", default="E:/ReAct-IR/checkpoints/backbone/best_backbone.pth")
    ap.add_argument("--image", default="", help="path to an input image for delta-map mode")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--volterra_rank", type=int, default=2)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--channels_last", type=int, default=1)

    return ap.parse_args()


def pick_ckpt_for_action(action_dir: str, ckpt_mode: str, epoch: int) -> str:
    if os.path.isfile(ckpt_mode) and ckpt_mode.lower().endswith(".pth"):
        return ckpt_mode

    if epoch > 0:
        pat = os.path.join(action_dir, f"epoch_{epoch:03d}_*.pth")
        files = glob.glob(pat)
        if not files:
            raise FileNotFoundError(f"No ckpt matches {pat}")
        return files[0]

    # latest
    return find_latest_ckpt(action_dir)


def main():
    args = parse_args()

    actions = [a.strip().lower() for a in args.actions.split(",") if a.strip()]
    valid = {"dedrop", "deblur", "desnow", "derain", "dehaze"}
    for a in actions:
        if a not in valid:
            raise ValueError(f"Invalid action '{a}'. Must be one of {sorted(list(valid))}.")

    # Load LoRA SDs
    action_to_ckpt: Dict[str, str] = {}
    action_to_sd: Dict[str, Dict[str, torch.Tensor]] = {}

    for a in actions:
        action_dir = os.path.join(args.lora_root, a)
        if not os.path.isdir(action_dir):
            raise FileNotFoundError(f"Missing action dir: {action_dir}")

        ckpt_path = pick_ckpt_for_action(action_dir, args.ckpt, args.epoch)
        action_to_ckpt[a] = ckpt_path
        action_to_sd[a] = load_lora_sd(ckpt_path, device="cpu")
        print(f"[Load] {a}: {ckpt_path}  | tensors={len(action_to_sd[a])}")

    # Weight-level comparisons (always)
    compare_actions_weight_level(action_to_sd, out_dir=args.out_dir, topk=args.topk)

    # Optional delta-map
    if int(args.do_delta) == 1:
        if not args.image or not os.path.isfile(args.image):
            print("\n[WARN] --do_delta=1 but --image is missing or not a file. Skip delta-map mode.")
            return

        project_root = args.project_root
        if project_root is None:
            # infer from lora_root parent
            project_root = os.path.abspath(os.path.join(args.lora_root, "..", ".."))
        run_delta_maps(
            project_root=project_root,
            backbone_ckpt=args.backbone_ckpt,
            action_to_lora_ckpt=action_to_ckpt,
            out_dir=args.out_dir,
            dim=args.dim,
            bias=args.bias,
            volterra_rank=args.volterra_rank,
            patch=args.patch,
            image_path=args.image,
            use_amp=args.use_amp,
            channels_last=args.channels_last,
        )


if __name__ == "__main__":
    main()


"""
========================
Example commands (PowerShell)
========================

1) Weight-level only (fast, no model import needed)
python -u e:/ReAct-IR/tools/compare_lora_actions.py `
  --lora_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
  --actions "dehaze,deblur,derain,dedrop" `
  --ckpt latest `
  --out_dir "E:/ReAct-IR/results/lora_compare" `
  --topk 25

2) Compare a specific epoch across actions
python -u e:/ReAct-IR/tools/compare_lora_actions.py `
  --lora_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
  --actions "dehaze,deblur,derain,dedrop" `
  --epoch 5 `
  --out_dir "E:/ReAct-IR/results/lora_compare_ep005" `
  --topk 25

3) Weight + Delta-map (requires project modules importable)
python -u e:/ReAct-IR/tools/compare_lora_actions.py `
  --project_root "E:/ReAct-IR" `
  --lora_root "E:/ReAct-IR/checkpoints/toolbank_lora" `
  --actions "dehaze,deblur,derain,dedrop" `
  --ckpt latest `
  --do_delta 1 `
  --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/best_backbone.pth" `
  --image "E:/ReAct-IR/data/RESIDE-6K/test/hazy/0001.png" `
  --patch 256 --dim 64 --bias 0 --volterra_rank 2 `
  --out_dir "E:/ReAct-IR/results/lora_compare_with_delta"

Outputs:
- summary_pairs.csv
- pair_{a1}_vs_{a2}.csv (per-layer diffs)
- delta_maps/delta_{action}.png (if do_delta=1)
"""