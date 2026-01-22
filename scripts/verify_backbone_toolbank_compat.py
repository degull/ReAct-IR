# e:/ReAct-IR/scripts/verify_backbone_toolbank_compat.py
# ------------------------------------------------------------
# What this script verifies (your “확정” 2-step check):
#  1) STOP baseline:  VETNet(dim=?, bias=?, volterra_rank=?) loads backbone_ckpt with STRICT=True
#     -> no missing/unexpected/size mismatch
#  2) ToolBank load:  ToolBank(backbone=VETNet(...)) loads toolbank_ckpt with STRICT=True
#     -> no missing/unexpected/size mismatch
#
# Extra (recommended):
#  3) Compare key tensor shapes between backbone_ckpt and toolbank_ckpt backbone base
#     (patch_embed/output + a few internal keys). This confirms same architecture family.
#
# Usage (PowerShell):
#   python e:/ReAct-IR/scripts/verify_backbone_toolbank_compat.py `
#     --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
#     --toolbank_ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth" `
#     --dim 48 --bias 0 --volterra_rank 4 `
#     --strict 1 --show 30
# ------------------------------------------------------------

import os
import sys
import argparse
from typing import Dict, Any, Tuple, List

import torch
import yaml

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank, AdapterSpec
from models.planner.action_space import (
    A_DEDROP, A_DEBLUR, A_DESNOW, A_DERAIN, A_DEHAZE, A_STOP
)

ACTIONS_ALL = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE]


# =========================
# CKPT utilities
# =========================
def load_any_ckpt(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu")
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError while reading: {path}\n"
            f"- Windows에서 파일이 다른 프로세스(탐색기 미리보기/백업/동기화/열려있는 python)에서 잡고 있을 수 있습니다.\n"
            f"- 해당 파일을 사용하는 프로세스를 종료한 뒤 다시 실행하세요.\n"
            f"Original: {repr(e)}"
        )


def extract_state_dict(ckpt_obj: Any, preferred_keys: List[str]) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Returns (state_dict, container_name)
    - If ckpt is a dict and has one of preferred_keys, use it.
    - Else if it looks like a raw state_dict (all tensors), use it.
    """
    if isinstance(ckpt_obj, dict):
        for k in preferred_keys:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k], k

        # raw state_dict case
        all_tensor_like = True
        for _, v in ckpt_obj.items():
            if not torch.is_tensor(v):
                all_tensor_like = False
                break
        if all_tensor_like:
            return ckpt_obj, "<raw_state_dict>"

    raise RuntimeError(f"Cannot extract state_dict using keys={preferred_keys}. keys={list(ckpt_obj.keys())[:30]}")


def safe_load_strict(model: torch.nn.Module, sd: Dict[str, torch.Tensor], strict: bool = True) -> Tuple[bool, str]:
    """
    strict=True에서 RuntimeError(size mismatch 등)까지 잡아서
    사람이 보기 좋게 원인 요약을 반환.
    """
    try:
        ret = model.load_state_dict(sd, strict=strict)
        # strict=True면 ret은 missing/unexpected가 빈 리스트여야 정상
        missing = list(getattr(ret, "missing_keys", []))
        unexpected = list(getattr(ret, "unexpected_keys", []))
        ok = (len(missing) == 0 and len(unexpected) == 0)
        msg = f"missing={len(missing)} unexpected={len(unexpected)}"
        if not ok:
            msg += f"\n  missing_keys(sample)={missing[:20]}\n  unexpected_keys(sample)={unexpected[:20]}"
        return ok, msg
    except RuntimeError as e:
        return False, f"RuntimeError(load_state_dict): {str(e)}"


def infer_dim_in_out_from_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    pe = sd.get("patch_embed.weight", None)
    ow = sd.get("output.weight", None)
    out = {}
    if pe is not None:
        out["dim"] = int(pe.shape[0])
        out["in_channels"] = int(pe.shape[1])
        out["patch_embed.weight.shape"] = tuple(pe.shape)
    else:
        out["dim"] = None
        out["in_channels"] = None
        out["patch_embed.weight.shape"] = None

    if ow is not None:
        out["out_channels"] = int(ow.shape[0])
        out["output.weight.shape"] = tuple(ow.shape)
    else:
        out["out_channels"] = None
        out["output.weight.shape"] = None
    return out


# =========================
# ToolBank spec loader
# =========================
def load_tools_yaml_specs() -> Dict[str, AdapterSpec]:
    tools_yaml = os.path.join(PROJECT_ROOT, "configs", "tools.yaml")
    if not os.path.isfile(tools_yaml):
        raise FileNotFoundError(
            f"configs/tools.yaml not found at: {tools_yaml}\n"
            f"ToolBank를 training과 동일하게 구성하려면 tools.yaml이 필요합니다."
        )
    with open(tools_yaml, "r", encoding="utf-8") as f:
        cfg_tools = yaml.safe_load(f)

    raw_specs = (cfg_tools.get("toolbank", {}) or {}).get("adapters", {}) or {}
    adapter_specs: Dict[str, AdapterSpec] = {}
    for a in ACTIONS_ALL:
        s = raw_specs.get(a, None)
        if s is None:
            raise KeyError(f"tools.yaml missing adapter spec for action={a}")
        adapter_specs[a] = AdapterSpec(**s)
    return adapter_specs


# =========================
# Shape compare helper
# =========================
def find_key_by_contains(sd: Dict[str, torch.Tensor], must_contain: List[str]) -> str:
    keys = list(sd.keys())
    for k in keys:
        ok = True
        for token in must_contain:
            if token not in k:
                ok = False
                break
        if ok:
            return k
    return ""


def compare_some_backbone_shapes(backbone_sd: Dict[str, torch.Tensor], toolbank_sd: Dict[str, torch.Tensor], show: int = 20):
    """
    ToolBank ckpt는 보통 key prefix가 'backbone.'으로 시작.
    MultiLoRA 래퍼가 있으면 base 파라미터가 '...base.weight' 같은 형태일 수 있음.
    여기서는 "패턴 검색"으로 patch_embed/output 등 핵심 텐서 shape를 찾아 비교.
    """
    print("\n" + "-" * 90)
    print("[EXTRA] shape compare (backbone_ckpt vs toolbank_ckpt backbone base)")

    # backbone reference
    bb_pe = backbone_sd.get("patch_embed.weight", None)
    bb_ow = backbone_sd.get("output.weight", None)

    # toolbank: try find patch_embed weight candidates
    candidates = [
        (["patch_embed", "weight"], "patch_embed.weight"),
        (["patch_embed", "base", "weight"], "patch_embed.base.weight"),
        (["output", "weight"], "output.weight"),
        (["output", "base", "weight"], "output.base.weight"),
    ]

    rows = []
    for tokens, label in candidates:
        k = find_key_by_contains(toolbank_sd, tokens)
        if k:
            rows.append((label, k, tuple(toolbank_sd[k].shape)))
        else:
            rows.append((label, "<not_found>", None))

    print("[Backbone CKPT] patch_embed.weight =", None if bb_pe is None else tuple(bb_pe.shape))
    print("[Backbone CKPT] output.weight     =", None if bb_ow is None else tuple(bb_ow.shape))
    print("\n[ToolBank CKPT] candidates:")
    for label, k, shp in rows:
        print(f"  - {label:<22} key={k} shape={shp}")

    # also show some volterra hints
    hint_tokens = [
        ["volt", "weight"],
        ["volterra", "weight"],
    ]
    print("\n[ToolBank CKPT] volterra-ish key samples:")
    printed = 0
    for k in toolbank_sd.keys():
        kk = k.lower()
        if ("volt" in kk or "volterra" in kk) and ("weight" in kk or "bias" in kk):
            print("  -", k, tuple(toolbank_sd[k].shape) if torch.is_tensor(toolbank_sd[k]) else type(toolbank_sd[k]))
            printed += 1
            if printed >= show:
                break
    if printed == 0:
        print("  (none found by heuristic)")

    print("-" * 90)


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--toolbank_ckpt", required=True)

    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--bias", type=int, default=0)  # 0/1
    ap.add_argument("--volterra_rank", type=int, default=4)

    ap.add_argument("--strict", type=int, default=1)  # 0/1
    ap.add_argument("--show", type=int, default=30)
    return ap.parse_args()


def main():
    args = parse_args()
    strict = bool(int(args.strict))

    print("=" * 90)
    print("[INPUT]")
    print(" backbone_ckpt:", args.backbone_ckpt)
    print(" toolbank_ckpt:", args.toolbank_ckpt)
    print(f" expected VETNet: dim={args.dim} bias={bool(args.bias)} volterra_rank={args.volterra_rank}")
    print(" strict:", strict)
    print("=" * 90)

    # -------------------------
    # 1) STOP baseline check
    # -------------------------
    print("\n[1/2] STOP baseline (backbone only) strict load check")
    bb_ck = load_any_ckpt(args.backbone_ckpt)
    bb_sd, bb_container = extract_state_dict(bb_ck, preferred_keys=["state_dict"])
    bb_meta = infer_dim_in_out_from_sd(bb_sd)

    print(f"  ckpt container = {bb_container}")
    print(f"  inferred: dim={bb_meta['dim']} in_ch={bb_meta['in_channels']} out_ch={bb_meta['out_channels']}")
    if bb_meta["dim"] is None:
        raise RuntimeError("Cannot infer dim from backbone ckpt (missing patch_embed.weight)")

    model_bb = VETNet(dim=args.dim, bias=bool(args.bias), volterra_rank=args.volterra_rank)
    ok1, msg1 = safe_load_strict(model_bb, bb_sd, strict=strict)
    print("  load_state_dict:", "OK" if ok1 else "FAIL")
    print(" ", msg1)

    # additional: compare expected vs inferred
    if bb_meta["dim"] != args.dim:
        print(f"  [WARN] backbone_ckpt dim({bb_meta['dim']}) != expected dim({args.dim})")
    if bb_meta["in_channels"] != 3 or bb_meta["out_channels"] != 3:
        print(f"  [WARN] unusual in/out channels: in={bb_meta['in_channels']} out={bb_meta['out_channels']}")

    # -------------------------
    # 2) ToolBank strict load check
    # -------------------------
    print("\n[2/2] ToolBank strict load check")
    tb_ck = load_any_ckpt(args.toolbank_ckpt)
    tb_sd, tb_container = extract_state_dict(tb_ck, preferred_keys=["toolbank", "state_dict"])
    print(f"  ckpt container = {tb_container}")
    print(f"  state_dict keys = {len(tb_sd)}")

    adapter_specs = load_tools_yaml_specs()
    backbone_for_tb = VETNet(dim=args.dim, bias=bool(args.bias), volterra_rank=args.volterra_rank)

    device = torch.device("cpu")  # check only
    tb = ToolBank(
        backbone=backbone_for_tb,
        adapter_specs=adapter_specs,
        device=device,
        debug=False,
    ).to(device)

    ok2, msg2 = safe_load_strict(tb, tb_sd, strict=strict)
    print("  load_state_dict:", "OK" if ok2 else "FAIL")
    print(" ", msg2)

    # -------------------------
    # Extra shape compare (recommended)
    # -------------------------
    compare_some_backbone_shapes(backbone_sd=bb_sd, toolbank_sd=tb_sd, show=int(args.show))

    # -------------------------
    # Final verdict
    # -------------------------
    print("\n" + "=" * 90)
    if ok1 and ok2:
        print("[VERDICT] ✅ FULL COMPATIBILITY (as requested checks passed)")
        print(" - STOP baseline strict load: OK")
        print(" - ToolBank strict load:      OK")
    else:
        print("[VERDICT] ❌ NOT FULLY COMPATIBLE (one or more checks failed)")
        print(" - STOP baseline:", "OK" if ok1 else "FAIL")
        print(" - ToolBank load :", "OK" if ok2 else "FAIL")
    print("=" * 90)


if __name__ == "__main__":
    main()

"""
>>   --backbone_ckpt "E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth" `
>>   --toolbank_ckpt "E:/ReAct-IR/checkpoints/toolbank/epoch_050_loss0.0920.pth"
"""