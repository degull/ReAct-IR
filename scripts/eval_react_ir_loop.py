# E:/ReAct-IR/scripts/eval_react_ir_loop.py
# ------------------------------------------------------------
# ReAct-IR Evaluation (ToolBank-only, baselines, NO backbone_ckpt)
#
# Goal: "완전 고정/오류불가" single_action 동작 보장
#   - argparse 단계에서 mode/single_action 강제 검증 + 즉시 종료
#   - single_action을 action_space 상수로 정규화(normalize)해서 저장
#   - Mode/Tag/Actions를 1회 계산 후 "그 값만" 사용 (중간 덮어쓰기 방지)
#   - ToolBank 구성은 training과 동일하게 tools.yaml의 adapter_specs로 생성
#   - ckpt는 toolbank_ckpt 하나만 로드 (혼선 제거)
#
# Modes:
#   --mode fixed5 : [A_DERAIN, A_DEDROP, A_DEHAZE, A_DEBLUR, A_DESNOW]
#   --mode stop   : [A_STOP]
#   --mode single : [--single_action]
#
# Examples (PowerShell):
#   python e:/ReAct-IR/scripts/eval_react_ir_loop.py `
#     --toolbank_ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth" `
#     --preset rain100H `
#     --data_root "E:/ReAct-IR/data" `
#     --out_root  "E:/ReAct-IR/results/react_ir_eval_baselines" `
#     --mode single --single_action A_DESNOW `
#     --max_images 5 --progress_every 1
# ------------------------------------------------------------

import os, sys, argparse, time, math, glob
from typing import Optional, Dict, Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# Imports (project)
# ------------------------------------------------------------
from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank, AdapterSpec
from models.planner.action_space import (
    A_STOP, A_DEBLUR, A_DEDROP, A_DERAIN, A_DEHAZE, A_DESNOW
)

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Canonical action set (authoritative)
VALID_ACTIONS = {
    A_STOP: A_STOP,
    A_DERAIN: A_DERAIN,
    A_DEDROP: A_DEDROP,
    A_DEHAZE: A_DEHAZE,
    A_DEBLUR: A_DEBLUR,
    A_DESNOW: A_DESNOW,
}

def list_images(root: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(files)

def imread(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def imwrite(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)

def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(x).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)

def to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return (x * 255.0).round().astype(np.uint8)

def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_any_state_dict(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], str]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        if hasattr(ckpt, "keys"):
            return ckpt, "<root-obj>"
        raise ValueError(f"Unsupported checkpoint object type: {type(ckpt)}")

    for k in ["toolbank", "state_dict", "model", "net", "network", "backbone", "ema", "params"]:
        v = ckpt.get(k, None)
        if isinstance(v, dict) and len(v) > 0:
            return v, k

    if len(ckpt) > 0 and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, "<root-dict>"

    raise KeyError(f"No usable state_dict found in checkpoint. Keys={list(ckpt.keys())}")

def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    out: Dict[str, torch.Tensor] = {}
    plen = len(prefix)
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
        else:
            out[k] = v
    return out

def strip_known_wrappers(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = sd
    if any(k.startswith("module.") for k in out.keys()):
        out = strip_prefix(out, "module.")
    if any(k.startswith("toolbank.") for k in out.keys()):
        out = strip_prefix(out, "toolbank.")
    return out

def infer_dim_from_toolbank_sd(sd: Dict[str, torch.Tensor]) -> int:
    # backbone.patch_embed.weight: (dim, 3, 3, 3)
    for k, v in sd.items():
        if k.endswith("backbone.patch_embed.weight") and torch.is_tensor(v) and v.ndim == 4:
            dim = int(v.shape[0])
            print(f"[Infer] dim from '{k}' shape={tuple(v.shape)} => dim={dim}")
            return dim
    # backbone.output.weight: (3, dim, 3, 3)
    for k, v in sd.items():
        if k.endswith("backbone.output.weight") and torch.is_tensor(v) and v.ndim == 4:
            dim = int(v.shape[1])
            print(f"[Infer] dim from '{k}' shape={tuple(v.shape)} => dim={dim}")
            return dim
    raise RuntimeError("Failed to infer dim from toolbank_ckpt. Missing backbone.patch_embed.weight / backbone.output.weight.")

def normalize_action_string(s: str) -> str:
    """
    single_action이 'A_DESNOW', 'a_desnow', 'desnow' 등으로 와도
    항상 action_space 상수 문자열('A_DESNOW')로 정규화.
    """
    if s is None:
        return s
    t = str(s).strip()
    if t == "":
        return t

    # already canonical?
    if t in VALID_ACTIONS:
        return VALID_ACTIONS[t]

    # common variants: lowercase / no "A_" prefix
    t_up = t.upper()
    if t_up in VALID_ACTIONS:
        return VALID_ACTIONS[t_up]

    if not t_up.startswith("A_"):
        t_up2 = "A_" + t_up
        if t_up2 in VALID_ACTIONS:
            return VALID_ACTIONS[t_up2]

    return t  # invalid -> caller will error

# ------------------------------------------------------------
# Presets
# ------------------------------------------------------------
def build_presets(data_root: str):
    return {
        "rain100H": {
            "input": os.path.join(data_root, "rain100H/test/rain"),
            "gt":    os.path.join(data_root, "rain100H/test/norain"),
        },
        "rain100L": {
            "input": os.path.join(data_root, "rain100L/test/rain"),
            "gt":    os.path.join(data_root, "rain100L/test/norain"),
        },
        "csd": {
            "input": os.path.join(data_root, "CSD/Test/Snow"),
            "gt":    os.path.join(data_root, "CSD/Test/Gt"),
        },
        "day_raindrop": {
            "input": os.path.join(data_root, "DayRainDrop/Test"),
            "gt": None,
        },
        "night_raindrop": {
            "input": os.path.join(data_root, "NightRainDrop/Test"),
            "gt": None,
        },
        "reside6k": {
            "input": os.path.join(data_root, "RESIDE-6K/Test"),
            "gt": None,
        },
    }

# ------------------------------------------------------------
# Mode -> Actions (single_action "절대" 보장)
# ------------------------------------------------------------
def get_mode_actions_and_tag(mode: str, single_action: Optional[str]) -> Tuple[List[str], str]:
    mode = str(mode).strip().lower()

    if mode == "fixed5":
        actions = [A_DERAIN, A_DEDROP, A_DEHAZE, A_DEBLUR, A_DESNOW]
        return actions, "fixed5"

    if mode == "stop":
        actions = [A_STOP]
        return actions, "stop"

    if mode == "single":
        if single_action is None:
            raise ValueError("--mode single requires --single_action")
        sa = normalize_action_string(single_action)
        if sa not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid --single_action '{single_action}' (normalized='{sa}'). "
                f"Valid={sorted(list(VALID_ACTIONS.keys()))}"
            )
        actions = [sa]
        return actions, f"single_{sa}"

    raise ValueError(f"Unknown --mode '{mode}'. choices: fixed5/stop/single")

# ------------------------------------------------------------
# Build adapter_specs exactly like training (tools.yaml)
# ------------------------------------------------------------
def build_adapter_specs_from_tools_yaml() -> Dict[str, AdapterSpec]:
    tools_path = os.path.join(PROJECT_ROOT, "configs", "tools.yaml")
    if not os.path.isfile(tools_path):
        raise FileNotFoundError(f"tools.yaml not found: {tools_path}")

    cfg_tools = load_yaml(tools_path)
    if "toolbank" not in cfg_tools or "adapters" not in cfg_tools["toolbank"]:
        raise KeyError("configs/tools.yaml must contain: toolbank: adapters: ...")

    raw_specs = cfg_tools["toolbank"]["adapters"]

    actions_all = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE]
    adapter_specs: Dict[str, AdapterSpec] = {}
    for a in actions_all:
        s = raw_specs.get(a, None)
        if s is None:
            raise KeyError(f"tools.yaml missing adapter spec for action '{a}'")
        adapter_specs[a] = AdapterSpec(**s)

    return adapter_specs

# ------------------------------------------------------------
# Args + "오류불가" 검증
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolbank_ckpt", required=True)

    ap.add_argument("--preset", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_root", default=None)

    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--out_dir", default=None)

    ap.add_argument("--mode", default="fixed5", choices=["fixed5", "stop", "single"])
    ap.add_argument("--single_action", default=None)

    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--volterra_rank", type=int, default=4, help="Must match training (train_toolbank.py uses 4).")
    ap.add_argument("--runtime_scale", type=float, default=1.0)
    ap.add_argument("--max_images", type=int, default=0, help="If >0, limit number of images per preset.")
    ap.add_argument("--progress_every", type=int, default=10, help="Print progress every N images (0 disables).")
    ap.add_argument("--print_args", action="store_true", help="Print resolved args/actions (debug).")

    args = ap.parse_args()

    # Hard validation: single_action must be valid when mode=single
    if args.mode == "single":
        if args.single_action is None:
            ap.error("--mode single requires --single_action")
        # normalize and validate here, then write back (so later code cannot 'sometimes' differ)
        sa = normalize_action_string(args.single_action)
        if sa not in VALID_ACTIONS:
            ap.error(
                f"Invalid --single_action '{args.single_action}' (normalized='{sa}'). "
                f"Valid={sorted(list(VALID_ACTIONS.keys()))}"
            )
        args.single_action = sa  # <-- 확정(고정)

    # Basic path validation
    if not os.path.isfile(args.toolbank_ckpt):
        ap.error(f"--toolbank_ckpt not found: {args.toolbank_ckpt}")

    if args.preset is not None:
        if args.data_root is None or args.out_root is None:
            ap.error("--preset requires --data_root and --out_root")

    return args

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
def main():
    args = parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("[Device]", device)

    # Resolve mode once (and ONLY use these values)
    actions, tag = get_mode_actions_and_tag(args.mode, args.single_action)

    if args.print_args:
        print("[ARGS] mode=", args.mode, "single_action=", args.single_action)
        print("[ARGS] resolved tag=", tag, "actions=", actions)

    # Load ckpt state_dict
    sd_raw, key = load_any_state_dict(args.toolbank_ckpt)
    sd_raw = strip_known_wrappers(sd_raw)
    print(f"[CKPT] ToolBank ckpt container='{key}' keys={len(sd_raw)}")

    # Infer dim (or override)
    if args.dim is not None:
        dim = int(args.dim)
        print(f"[Infer] dim override => {dim}")
    else:
        dim = infer_dim_from_toolbank_sd(sd_raw)

    # Build ToolBank exactly like training
    adapter_specs = build_adapter_specs_from_tools_yaml()

    backbone = VETNet(dim=dim, volterra_rank=int(args.volterra_rank)).to(device).eval()
    toolbank = ToolBank(
        backbone=backbone,
        adapter_specs=adapter_specs,
        device=device,
        debug=False,
    ).to(device).eval()

    # Load (should be clean if specs match training)
    missing, unexpected = toolbank.load_state_dict(sd_raw, strict=False)
    print(f"[Load] toolbank.load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[Mode] mode={args.mode} tag={tag} actions={actions} runtime_scale={float(args.runtime_scale)}")

    # Preset mode
    if args.preset is not None:
        presets = build_presets(args.data_root)
        keys = list(presets.keys()) if args.preset == "all" else [args.preset]

        for name in keys:
            if name not in presets:
                raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

            cfg = presets[name]
            out_dir = os.path.join(args.out_root, tag, name)
            run_one(
                name=f"{name}({tag})",
                input_dir=cfg["input"],
                gt_dir=cfg["gt"],
                out_dir=out_dir,
                toolbank=toolbank,
                device=device,
                actions=actions,
                save_intermediate=bool(args.save_intermediate),
                runtime_scale=float(args.runtime_scale),
                max_images=int(args.max_images),
                progress_every=int(args.progress_every),
            )
        return

    # Single dir mode
    if args.input_dir is None or args.out_dir is None:
        raise ValueError("Single dir mode requires --input_dir and --out_dir")

    out_dir = os.path.join(args.out_dir, tag)
    run_one(
        name=f"single({tag})",
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        out_dir=out_dir,
        toolbank=toolbank,
        device=device,
        actions=actions,
        save_intermediate=bool(args.save_intermediate),
        runtime_scale=float(args.runtime_scale),
        max_images=int(args.max_images),
        progress_every=int(args.progress_every),
    )

def run_one(
    name: str,
    input_dir: str,
    gt_dir: Optional[str],
    out_dir: str,
    toolbank: ToolBank,
    device: torch.device,
    actions: List[str],
    save_intermediate: bool,
    runtime_scale: float,
    max_images: int,
    progress_every: int,
):
    print(f"\n=== [{name}] ===")
    if input_dir is None or (not os.path.isdir(input_dir)):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    imgs = list_images(input_dir)
    if max_images > 0:
        imgs = imgs[:max_images]
    print(f"Found {len(imgs)} images")

    os.makedirs(out_dir, exist_ok=True)

    # GT map
    gt_map: Dict[str, str] = {}
    if gt_dir is not None and os.path.isdir(gt_dir):
        gt_imgs = list_images(gt_dir)
        gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_imgs}
        print(f"Found {len(gt_imgs)} GT images")

    sum_psnr = 0.0
    n_psnr = 0
    t0 = time.time()

    toolbank.eval()
    with torch.no_grad():
        for i, p in enumerate(imgs, 1):
            x = to_tensor(imread(p), device=device)
            base = os.path.splitext(os.path.basename(p))[0]

            inter_dir = None
            if save_intermediate:
                inter_dir = os.path.join(out_dir, "intermediate", base)
                os.makedirs(inter_dir, exist_ok=True)

            for step_idx, act in enumerate(actions, 1):
                # ToolBank training code uses: pred = toolbank.apply(x, action)
                # Here we use runtime_scale explicitly.
                x = toolbank.apply(x, act, runtime_scale=runtime_scale)

                if inter_dir is not None:
                    imwrite(
                        os.path.join(inter_dir, f"step{step_idx:02d}_{str(act)}.png"),
                        to_uint8(x),
                    )

            imwrite(os.path.join(out_dir, f"{base}.png"), to_uint8(x))

            if gt_map:
                gtp = gt_map.get(base, None)
                if gtp is not None:
                    gt = to_tensor(imread(gtp), device=device)
                    sum_psnr += psnr(x, gt)
                    n_psnr += 1

            if (progress_every > 0) and ((i % progress_every) == 0 or i == len(imgs)):
                dt = time.time() - t0
                print(f"[{name}] {i}/{len(imgs)} done | elapsed={dt/60:.1f}m")

    print(f"[Saved] {out_dir}")
    if n_psnr > 0:
        print(f"[{name}] PSNR(avg over {n_psnr}) = {sum_psnr / n_psnr:.4f} dB")
    else:
        print(f"[{name}] PSNR skipped (no GT)")

if __name__ == "__main__":
    main()


# STOP (LoRA off)
"""
python e:/ReAct-IR/scripts/eval_react_ir_loop.py `
  --toolbank_ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth" `
  --preset rain100H --data_root "E:/ReAct-IR/data" --out_root "E:/ReAct-IR/results/react_ir_eval_baselines" `
  --mode stop --progress_every 50

"""

# Best-single-action 찾기 (이미 5장으로는 대충 봤고, 이제 100장)
"""
# 예: DESNOW
python e:/ReAct-IR/scripts/eval_react_ir_loop.py `
  --toolbank_ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth" `
  --preset rain100H --data_root "E:/ReAct-IR/data" --out_root "E:/ReAct-IR/results/react_ir_eval_baselines" `
  --mode single --single_action A_DESNOW --progress_every 50

"""

# fixed5
"""
python e:/ReAct-IR/scripts/eval_react_ir_loop.py `
  --toolbank_ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth" `
  --preset rain100H --data_root "E:/ReAct-IR/data" --out_root "E:/ReAct-IR/results/react_ir_eval_baselines" `
  --mode fixed5 --progress_every 50

"""