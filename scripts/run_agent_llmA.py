# scripts/run_agent_llmA.py
# ------------------------------------------------------------
# Rule-based Agent Loop (LLM-A style) for ReAct-IR
# - Backbone robust load: key remap + shape-adapt (norm vector -> (1,C,1,1))
# ------------------------------------------------------------
import os
import sys
import glob
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Project imports
# --------------------------------------------------
from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def pil_load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return x


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = x.clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
    return Image.fromarray(arr)


def strip_module_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        kk = k[7:] if k.startswith("module.") else k
        out[kk] = v
    return out


def pick_latest_ckpt(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    cands = sorted(glob.glob(os.path.join(folder, "*.pth")))
    if not cands:
        return None
    best = [p for p in cands if "best" in os.path.basename(p).lower()]
    if best:
        return best[-1]
    cands = sorted(cands, key=lambda p: os.path.getmtime(p))
    return cands[-1]


def list_images(root_or_file: str) -> List[str]:
    if os.path.isfile(root_or_file):
        return [root_or_file]
    out = []
    for dp, _, fnames in os.walk(root_or_file):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(dp, f))
    out.sort()
    return out


# ============================================================
# Diagnoser importer + build
# ============================================================
def try_import_train_diagnoser_map(debug: bool = False):
    try:
        import importlib
        m = importlib.import_module("scripts.train_diagnoser_map")
        if debug:
            print("[DEBUG][Import] module ok: scripts.train_diagnoser_map")
        return m
    except Exception as e:
        if debug:
            print("[DEBUG][Import] failed:", repr(e))
        return None


@torch.no_grad()
def resize_pos_embed_square_with_cls(pos_embed_cls: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    assert pos_embed_cls.dim() == 3 and pos_embed_cls.size(0) == 1
    cls = pos_embed_cls[:, :1]
    tok = pos_embed_cls[:, 1:]
    D = tok.size(-1)
    tok = tok.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2).contiguous()
    tok2 = F.interpolate(tok, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
    tok2 = tok2.permute(0, 2, 3, 1).contiguous().reshape(1, new_grid * new_grid, D)
    return torch.cat([cls, tok2], dim=1)


def build_diagnoser(ckpt_path: str, img_size: int, device: torch.device, debug: bool):
    m = try_import_train_diagnoser_map(debug=debug)
    if m is None:
        raise RuntimeError("Cannot import scripts.train_diagnoser_map. Check file path.")

    actions_internal = getattr(m, "ACTIONS_INTERNAL", None)
    if actions_internal is None:
        raise RuntimeError("scripts.train_diagnoser_map.py missing ACTIONS_INTERNAL")
    actions = list(actions_internal)

    cls = getattr(m, "TinyViTDiagnoserMap", None)
    if cls is None:
        raise RuntimeError("scripts.train_diagnoser_map.py missing TinyViTDiagnoserMap")

    model = cls(img_size=img_size, patch_size=16, embed_dim=384, depth=6, num_heads=6, num_labels=5)
    model.to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = strip_module_prefix(sd)
    sd2 = dict(sd)

    if ("map_head.weight" in sd2) and ("head_map.weight" not in sd2):
        sd2["head_map.weight"] = sd2.pop("map_head.weight")
    if ("map_head.bias" in sd2) and ("head_map.bias" not in sd2):
        sd2["head_map.bias"] = sd2.pop("map_head.bias")

    if "head_map.weight" in sd2 and isinstance(sd2["head_map.weight"], torch.Tensor):
        w = sd2["head_map.weight"]
        if w.ndim == 2:
            if debug:
                print("[DEBUG][Diagnoser] head_map.weight: reshape (5,D) -> (5,D,1,1)")
            sd2["head_map.weight"] = w[:, :, None, None].contiguous()

    if "pos_embed" in sd2 and isinstance(sd2["pos_embed"], torch.Tensor):
        pe = sd2["pos_embed"]
        cur_pe = model.pos_embed.detach()
        if pe.shape != cur_pe.shape:
            if pe.dim() == 3 and cur_pe.dim() == 3 and pe.shape[0] == 1 and cur_pe.shape[0] == 1:
                if pe.shape[1] + 1 == cur_pe.shape[1] and pe.shape[2] == cur_pe.shape[2]:
                    if debug:
                        print("[DEBUG][Diagnoser] pos_embed: add CLS (ckpt has no CLS)")
                    cls_tok = torch.zeros((1, 1, pe.shape[2]), dtype=pe.dtype)
                    sd2["pos_embed"] = torch.cat([cls_tok, pe], dim=1)
                else:
                    n_old = int(pe.shape[1] - 1)
                    n_new = int(cur_pe.shape[1] - 1)
                    g_old = int(round(np.sqrt(max(1, n_old))))
                    g_new = int(round(np.sqrt(max(1, n_new))))
                    if g_old * g_old == n_old and g_new * g_new == n_new and pe.shape[2] == cur_pe.shape[2]:
                        if debug:
                            print(f"[DEBUG][Diagnoser] resize pos_embed: old_grid={g_old} -> new_grid={g_new}")
                        sd2["pos_embed"] = resize_pos_embed_square_with_cls(pe, old_grid=g_old, new_grid=g_new)
                    else:
                        sd2.pop("pos_embed", None)

    missing, unexpected = model.load_state_dict(sd2, strict=False)
    if debug:
        print(f"[DEBUG][Diagnoser] load strict=False | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("[DEBUG][Diagnoser] missing(sample):", list(missing)[:10])
        if unexpected:
            print("[DEBUG][Diagnoser] unexpected(sample):", list(unexpected)[:10])

    return model, actions


@torch.no_grad()
def run_diagnoser_map_vec(model: nn.Module, x: torch.Tensor, debug: bool = False) -> np.ndarray:
    x = x.float()
    device_type = "cuda" if x.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        logits_g, logits_m = model(x)

    m = torch.sigmoid(logits_m)[0]   # (5,h,w)
    sev_mean = m.mean(dim=(1, 2))    # (5,)

    h, w = m.shape[-2], m.shape[-1]
    k = max(1, int(0.05 * h * w))
    flat = m.view(5, -1)
    topk_vals, _ = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
    sev_topk = topk_vals.mean(dim=1)
    sev = torch.maximum(sev_mean, sev_topk)

    if debug:
        print("[DEBUG][Diagnoser] sev_mean(map):", sev_mean.detach().cpu().numpy()[None, :])
        print("[DEBUG][Diagnoser] sev_topk(map):", sev_topk.detach().cpu().numpy()[None, :])
        print("[DEBUG][Diagnoser] sev(final):", sev.detach().cpu().numpy()[None, :])

    return sev.detach().cpu().numpy().astype(np.float32)


# ============================================================
# ToolBank helpers
# ============================================================
def _extract_state_dict_from_ckpt(ckpt_obj: Any) -> Dict[str, Any]:
    if isinstance(ckpt_obj, dict):
        sd = ckpt_obj.get("lora_state_dict", None)
        if sd is None:
            sd = ckpt_obj.get("state_dict", None)
        if sd is None:
            sd = ckpt_obj.get("model", None)
        if sd is None:
            sd = ckpt_obj
    else:
        sd = ckpt_obj
    if not isinstance(sd, dict):
        raise RuntimeError("checkpoint does not contain a dict state_dict")
    return strip_module_prefix(sd)


def infer_lora_rank_from_ckpt(ckpt_path: str, action: str, debug: bool = False) -> Optional[int]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_state_dict_from_ckpt(ckpt)
        needle = f".lora_A.{action}.weight"
        for k, v in sd.items():
            if (needle in k) and isinstance(v, torch.Tensor):
                if v.ndim == 4:
                    r = int(v.shape[0])
                    if debug:
                        print(f"[DEBUG][LoRA] inferred rank from {os.path.basename(ckpt_path)} key={k} -> r={r}")
                    return r
                if v.ndim == 2:
                    r = int(v.shape[0])
                    return r
        return None
    except Exception as e:
        if debug:
            print("[DEBUG][LoRA] infer rank failed:", repr(e))
        return None


def load_toolbank_lora_for_action(tb: ToolBank, action: str, ckpt_path: str, debug: bool):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict_from_ckpt(ckpt)
    info = tb.load_lora_state_dict_for_action(action, sd, strict=False, map_location="cpu")
    if debug:
        print(f"[DEBUG][LoRA({action})] load info:", info)
    return info


def preload_all_loras(lora_root: str, actions: List[str]) -> Dict[str, str]:
    act2dir = {
        "A_DEBLUR": "deblur",
        "A_DERAIN": "derain",
        "A_DESNOW": "desnow",
        "A_DEHAZE": "dehaze",
        "A_DEDROP": "dedrop",
    }
    loaded = {}
    for a in actions:
        d = act2dir.get(a, None)
        if d is None:
            continue
        ckpt = pick_latest_ckpt(os.path.join(lora_root, d))
        if ckpt is None:
            print(f"[ToolBank][WARN] no ckpt found for {a} in {lora_root}/{d}")
            continue
        loaded[a] = ckpt
        print(f"[LoRA] {a} -> {ckpt}")
    return loaded


# ============================================================
# ✅ Backbone robust loader: key remap + shape-adapt
# ============================================================
def remap_backbone_key(k: str) -> List[str]:
    kk = k
    kk = kk.replace(".blocks.", ".body.")
    kk = kk.replace(".gdfn.", ".ffn.")

    if ".volterra." in kk:
        # drop extra ranks if old ckpt had more (2,3)
        if ".W2a.2." in kk or ".W2a.3." in kk or ".W2b.2." in kk or ".W2b.3." in kk:
            return []
        kk1 = kk.replace(".volterra.", ".volt1.")
        kk2 = kk.replace(".volterra.", ".volt2.")
        return [kk1, kk2]

    return [kk]


def remap_backbone_state_dict(sd: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        cands = remap_backbone_key(k)
        for kk in cands:
            if kk and kk not in out:
                out[kk] = v
    if debug:
        print(f"[BackboneRemap] in_keys={len(sd)} out_keys={len(out)}")
    return out


def adapt_tensor_to_target(v: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Try to adapt ckpt tensor v to match target shape.
    Main fix: norm params (C,) -> (1,C,1,1)
    """
    if not isinstance(v, torch.Tensor) or not isinstance(target, torch.Tensor):
        return None

    if v.shape == target.shape:
        return v

    # (C,) -> (1,C,1,1)
    if v.ndim == 1 and target.ndim == 4:
        C = v.shape[0]
        if target.shape[0] == 1 and target.shape[2] == 1 and target.shape[3] == 1 and target.shape[1] == C:
            return v.view(1, C, 1, 1).contiguous()

    # (1,C) or (C,1) -> (1,C,1,1) (rare)
    if v.ndim == 2 and target.ndim == 4 and target.shape[0] == 1 and target.shape[2] == 1 and target.shape[3] == 1:
        if v.shape[0] == 1 and v.shape[1] == target.shape[1]:
            return v.view(1, target.shape[1], 1, 1).contiguous()
        if v.shape[1] == 1 and v.shape[0] == target.shape[1]:
            return v.view(1, target.shape[1], 1, 1).contiguous()

    return None


def load_state_dict_shape_safe_adaptive(
    model: nn.Module,
    sd_in: Dict[str, Any],
    debug: bool = True,
) -> Tuple[float, Dict[str, int]]:
    """
    Load keys that exist and either match shape or can be adapted.
    Coverage computed over intersection keys (matchable).
    """
    msd = model.state_dict()

    matchable = []
    for k, v in sd_in.items():
        if k in msd and isinstance(v, torch.Tensor) and isinstance(msd[k], torch.Tensor):
            matchable.append(k)

    ok = {}
    adapted = []
    skipped_shape = []
    for k in matchable:
        v = sd_in[k]
        t = msd[k]
        if v.shape == t.shape:
            ok[k] = v
        else:
            vv = adapt_tensor_to_target(v, t)
            if vv is not None and vv.shape == t.shape:
                ok[k] = vv
                adapted.append(k)
            else:
                skipped_shape.append(k)

    missing, unexpected = model.load_state_dict(ok, strict=False)

    matchable_n = len(matchable)
    loaded_n = len(ok)
    coverage = loaded_n / max(1, matchable_n)

    stats = {
        "model_keys": len(msd),
        "ckpt_keys": len(sd_in),
        "matchable": matchable_n,
        "loaded": loaded_n,
        "adapted": len(adapted),
        "skipped_shape": len(skipped_shape),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }

    print(
        "[BackboneLoad] "
        f"model_keys={stats['model_keys']} ckpt_keys={stats['ckpt_keys']} "
        f"matchable={stats['matchable']} loaded={stats['loaded']} adapted={stats['adapted']} "
        f"skipped_shape={stats['skipped_shape']} coverage={coverage:.4f} "
        f"missing={stats['missing']} unexpected={stats['unexpected']}"
    )
    if debug:
        if adapted:
            print("[BackboneLoad][DEBUG] adapted(sample):", adapted[:20])
        if skipped_shape:
            print("[BackboneLoad][DEBUG] skipped_shape(sample):", skipped_shape[:20])
        if missing:
            print("[BackboneLoad][DEBUG] missing(sample):", list(missing)[:20])
        if unexpected:
            print("[BackboneLoad][DEBUG] unexpected(sample):", list(unexpected)[:20])

    return coverage, stats


# ============================================================
# Policy helpers
# ============================================================
def all_below(vec: np.ndarray, thr: float) -> bool:
    return bool(np.all(vec <= thr))


def choose_action_max_sev(
    vec: np.ndarray,
    actions: List[str],
    last_action: Optional[str],
    last_vec: Optional[np.ndarray],
    no_improve_count: int,
    no_improve_eps: float,
    no_improve_patience: int,
    choose_thr: float,
    debug: bool = False,
):
    order = list(np.argsort(-vec))
    best_idx = int(order[0])
    best_action = actions[best_idx]

    if last_action is not None and last_vec is not None:
        a_idx = actions.index(last_action)
        delta = float(last_vec[a_idx] - vec[a_idx])
        if delta < no_improve_eps:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if no_improve_count >= no_improve_patience:
            switched = False
            for j in order:
                cand_idx = int(j)
                cand_action = actions[cand_idx]
                if cand_action == last_action:
                    continue
                if float(vec[cand_idx]) > choose_thr:
                    best_action = cand_action
                    best_idx = cand_idx
                    no_improve_count = 0
                    switched = True
                    if debug:
                        print(f"[DEBUG][Policy] no-improve switch: {last_action} -> {best_action} (sev={vec[cand_idx]:.3f})")
                    break
            if not switched and debug:
                print("[DEBUG][Policy] no-improve but no valid alternative above thresh -> keep best")

    return best_action, best_idx, no_improve_count


def choose_scale(sev: float, scales: List[float], mode: str = "proportional") -> float:
    if mode == "fixed1":
        return 1.0
    if mode == "proportional":
        target = float(np.clip(sev, 0.0, 1.0))
        return float(min(scales, key=lambda s: abs(float(s) - target)))
    return float(max(scales))


def apply_stop_bias(sev: np.ndarray, bias: np.ndarray) -> np.ndarray:
    v = sev - bias
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    return v


# ============================================================
# Args
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--toolbank_lora_root", required=True)

    ap.add_argument("--diagnoser_ckpt", required=True)

    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])

    ap.add_argument("--stop_vec_thresh", type=float, default=0.10)
    ap.add_argument("--stop_bias", type=float, nargs=5, default=None)

    ap.add_argument("--save_intermediate", type=int, default=1)
    ap.add_argument("--debug", type=int, default=0)

    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--volterra_rank", type=int, default=2)
    ap.add_argument("--use_amp", type=int, default=1)

    ap.add_argument("--toolbank_rank", type=int, default=-1)
    ap.add_argument("--toolbank_alpha", type=float, default=1.0)
    ap.add_argument("--wrap_only_1x1", type=int, default=1)

    ap.add_argument("--policy", default="max_sev", choices=["max_sev"])
    ap.add_argument("--scale_mode", default="proportional", choices=["proportional", "fixed1"])
    ap.add_argument("--no_improve_eps", type=float, default=0.01)
    ap.add_argument("--no_improve_patience", type=int, default=2)
    ap.add_argument("--choose_thr", type=float, default=0.10)
    ap.add_argument("--min_apply_scale", type=float, default=0.25)

    ap.add_argument("--bb_min_coverage", type=float, default=0.90)
    ap.add_argument("--force_run", type=int, default=0)

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    debug = bool(args.debug)

    safe_makedirs(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[Input]", args.input)
    print(f"[Policy] policy={args.policy}, scale_mode={args.scale_mode} (MAP severity used for decision)")

    img0 = pil_load_rgb(args.input)
    x0 = pil_to_tensor(img0).unsqueeze(0)
    x = x0.to(device)

    print("[Input] x shape =", tuple(x.shape))

    # ---- backbone
    print("[Backbone]", args.backbone_ckpt)
    backbone = VETNet(dim=int(args.dim), bias=bool(int(args.bias)), volterra_rank=int(args.volterra_rank)).to(device)

    ckpt_bb = torch.load(args.backbone_ckpt, map_location="cpu")
    sd_bb = ckpt_bb.get("state_dict", ckpt_bb) if isinstance(ckpt_bb, dict) else ckpt_bb
    sd_bb = strip_module_prefix(sd_bb)

    sd_bb2 = remap_backbone_state_dict(sd_bb, debug=debug)
    coverage, stats = load_state_dict_shape_safe_adaptive(backbone, sd_bb2, debug=debug)

    if coverage < float(args.bb_min_coverage) and int(args.force_run) == 0:
        raise RuntimeError(
            f"Backbone weights NOT properly loaded (coverage={coverage:.2f} < {args.bb_min_coverage:.2f}).\n"
            f"Checkpoint and current VETNet still differ.\n"
            f"Try another backbone ckpt OR run with --force_run 1 (not recommended)."
        )
    if coverage < float(args.bb_min_coverage) and int(args.force_run) == 1:
        print(f"[BackboneLoad][WARN] low coverage={coverage:.2f} but continuing due to --force_run 1")

    backbone.eval()

    # ---- diagnoser
    diagnoser_img_size = 256
    diagnoser, actions = build_diagnoser(
        ckpt_path=args.diagnoser_ckpt,
        img_size=diagnoser_img_size,
        device=device,
        debug=debug,
    )
    print("[ACTIONS]", actions)
    print(f"[Diagnoser] ckpt={args.diagnoser_ckpt} | type=vit_map | img_size={diagnoser_img_size}")

    # ---- toolbank
    act2dir = {"A_DEBLUR": "deblur", "A_DERAIN": "derain", "A_DESNOW": "desnow", "A_DEHAZE": "dehaze", "A_DEDROP": "dedrop"}
    tmp_lora_paths = {}
    for a in actions:
        d = act2dir.get(a, None)
        if d is None:
            continue
        ckpt = pick_latest_ckpt(os.path.join(args.toolbank_lora_root, d))
        if ckpt is not None:
            tmp_lora_paths[a] = ckpt

    tb_rank = int(args.toolbank_rank)
    if tb_rank <= 0:
        inferred = None
        for a in actions:
            if a in tmp_lora_paths:
                inferred = infer_lora_rank_from_ckpt(tmp_lora_paths[a], a, debug=debug)
                if inferred is not None:
                    tb_rank = int(inferred)
                    break
        if tb_rank <= 0:
            tb_rank = 4
            print(f"[ToolBank][WARN] cannot infer LoRA rank -> fallback rank={tb_rank}")
        else:
            print(f"[ToolBank] auto-inferred lora_rank={tb_rank}")

    tb = ToolBank(
        backbone=backbone,
        actions=actions,
        rank=tb_rank,
        alpha=float(args.toolbank_alpha),
        wrap_only_1x1=bool(int(args.wrap_only_1x1)),
    ).to(device)
    print("[ToolBank] injected. lora_root(for ckpt)=", args.toolbank_lora_root)
    print(f"[ToolBank] lora_rank={tb.rank}, alpha={tb.alpha}, wrap_only_1x1={int(tb.wrap_only_1x1)}")

    lora_paths = preload_all_loras(args.toolbank_lora_root, actions)

    # ---- stop bias
    if args.stop_bias is not None:
        stop_bias = np.array(list(map(float, args.stop_bias)), dtype=np.float32)
        print("[StopBias] mode=manual")
        print("[StopBias] bias =", stop_bias.tolist())
    else:
        stop_bias = np.zeros((len(actions),), dtype=np.float32)
        print("[StopBias] mode=none (zeros)")

    last_action = None
    last_vec = None
    no_improve_count = 0

    x_cur = x.clone().float()
    tensor_to_pil(x_cur[0]).save(os.path.join(args.out_dir, "step_-1_input.png"))

    for step in range(int(args.max_steps)):
        print(f"\n=== Step {step} ===")

        x_cur = torch.nan_to_num(x_cur, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

        x_diag = F.interpolate(x_cur.float(), size=(diagnoser_img_size, diagnoser_img_size), mode="bilinear", align_corners=False)
        sev = run_diagnoser_map_vec(diagnoser, x_diag, debug=debug)
        stop_vec = apply_stop_bias(sev, stop_bias)

        if debug:
            print("[DEBUG][Agent] sev_map:", sev[None, :])
            print("[DEBUG][Agent] stop_vec(sev-bias):", stop_vec[None, :])

        if all_below(stop_vec, float(args.stop_vec_thresh)):
            decision = {"action": "A_STOP", "scale": 0.0, "stop": True, "reason": "stop_vec below threshold"}
            print("Decision:", decision)
            break

        best_action, best_idx, no_improve_count = choose_action_max_sev(
            vec=stop_vec,
            actions=actions,
            last_action=last_action,
            last_vec=last_vec,
            no_improve_count=no_improve_count,
            no_improve_eps=float(args.no_improve_eps),
            no_improve_patience=int(args.no_improve_patience),
            choose_thr=float(args.choose_thr),
            debug=debug,
        )
        sev_best = float(stop_vec[best_idx])
        scale = choose_scale(sev_best, list(map(float, args.scales)), mode=str(args.scale_mode))
        if float(scale) < float(args.min_apply_scale) and sev_best > float(args.stop_vec_thresh):
            scale = float(args.min_apply_scale)

        decision = {
            "action": best_action,
            "scale": float(scale),
            "stop": False,
            "reason": f"max-severity: {best_action} (sev={sev_best:.3f}) | scale={scale:.2f}",
        }
        print("Decision:", decision)

        if best_action in lora_paths:
            ckpt_path = lora_paths[best_action]
            _ = load_toolbank_lora_for_action(tb, best_action, ckpt_path, debug=debug)
        else:
            print(f"[ToolBank][WARN] no lora ckpt for action={best_action}")

        tb.activate(best_action, float(scale))
        tb.eval()

        with torch.no_grad():
            if device.type == "cuda" and int(args.use_amp) == 1:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    y = tb(x_cur)
                x_cur = y.float()
            else:
                x_cur = tb(x_cur).float()

        x_cur = torch.nan_to_num(x_cur, nan=0.0, posinf=1.0, neginf=0.0)

        if int(args.save_intermediate) == 1:
            out_path = os.path.join(args.out_dir, f"step_{step:02d}_{best_action}_{scale:.2f}.png")
            tensor_to_pil(x_cur[0]).save(out_path)
            if debug:
                print("[DEBUG] saved", out_path)

        last_action = best_action
        last_vec = stop_vec.copy()

    out_final = os.path.join(args.out_dir, "final.png")
    tensor_to_pil(x_cur[0]).save(out_final)
    print("\n[Done] final saved:", out_final)


if __name__ == "__main__":
    main()
