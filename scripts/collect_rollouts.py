# Diagnoser가 낸 상태(s₀, m₀ 등)에서, 각 LoRA(action)를 여러 scale로 적용해보고
# ΔPSNR/ΔSSIM(개선량) 을 기록해서 “어떤 행동이 이득이었는지” 데이터셋을 만든다.

import os
import sys
import json
import time
import math
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank

from datasets.lora_dataset import (
    ActionPairConfig,
    build_action_pairs,
)

# --------------------------------------------------
# Optional skimage SSIM
# --------------------------------------------------
try:
    from skimage.metrics import structural_similarity as _ssim
    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False


# ============================================================
# Constants
# ============================================================
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
ACTIONS_INTERNAL = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
IDX_TO_ACTION = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]


# ============================================================
# Utils
# ============================================================
def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_tensor_chw(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


def pad_to_multiple(x: torch.Tensor, mul: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    x: (1,3,H,W)
    returns padded_x, pad=(left,right,top,bottom)
    """
    _, _, h, w = x.shape
    pad_h = (mul - (h % mul)) % mul
    pad_w = (mul - (w % mul)) % mul
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x = F.pad(x, (left, right, top, bottom), mode="reflect")
    return x, (left, right, top, bottom)


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    if left == right == top == bottom == 0:
        return x
    _, _, h, w = x.shape
    return x[:, :, top:h - bottom, left:w - right]


def psnr_torch(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> float:
    """
    pred, gt: (1,3,H,W) in [0,1]
    """
    mse = torch.mean((pred - gt) ** 2).item()
    mse = max(mse, eps)
    return float(10.0 * math.log10(1.0 / mse))


def ssim_optional(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    pred, gt: (1,3,H,W) in [0,1]
    """
    if not USE_SKIMAGE:
        return 0.0
    p = (pred[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    g = (gt[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return float(_ssim(g, p, channel_axis=2, data_range=255))


def parse_scales(scales: List[str]) -> List[float]:
    out = []
    for s in scales:
        if "," in s:
            for t in s.split(","):
                t = t.strip()
                if t:
                    out.append(float(t))
        else:
            out.append(float(s))
    return out


def choose_best_lora_ckpt(action_dir: str) -> str:
    """
    Picks best checkpoint for a given action folder.
    Priority:
      1) ckpt containing metrics.psnr (max)
      2) filename with '_Pxx.xx_' (max)
      3) newest mtime
    """
    files = sorted(glob.glob(os.path.join(action_dir, "*.pth")))
    if not files:
        raise FileNotFoundError(f"No .pth found under {action_dir}")

    best = None
    best_psnr = -1e9

    # Try load-metrics approach (safe-ish)
    for f in files:
        try:
            ckpt = torch.load(f, map_location="cpu")
            if isinstance(ckpt, dict):
                m = ckpt.get("metrics", None)
                if isinstance(m, dict) and "psnr" in m:
                    ps = float(m["psnr"])
                    if ps > best_psnr:
                        best_psnr = ps
                        best = f
        except Exception:
            pass

    if best is not None:
        return best

    # Try parse from filename _Pxx.xx_
    import re
    pat = re.compile(r"_P([0-9]+(\.[0-9]+)?)_")
    for f in files:
        m = pat.search(os.path.basename(f))
        if m:
            ps = float(m.group(1))
            if ps > best_psnr:
                best_psnr = ps
                best = f
    if best is not None:
        return best

    # fallback newest
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


# ============================================================
# DiagnoserMap model (must match your train_diagnoser_map.py)
# ============================================================
class TinyViTDiagnoserMap(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_labels: int = 5,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head_global = nn.Linear(embed_dim, num_labels)
        self.head_map = nn.Conv2d(embed_dim, num_labels, kernel_size=1, stride=1, padding=0, bias=True)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,3,H,W) with H=W=img_size
        B, C, H, W = x.shape
        t = self.patch_embed(x)                           # (B,D,g,g)
        t_flat = t.flatten(2).transpose(1, 2)            # (B,N,D)

        cls = self.cls_token.expand(B, -1, -1)           # (B,1,D)
        z = torch.cat([cls, t_flat], dim=1)              # (B,1+N,D)
        z = z + self.pos_embed[:, : z.size(1)]
        z = self.pos_drop(z)

        z = self.encoder(z)                              # (B,1+N,D)

        g = self.norm(z[:, 0])                           # (B,D)
        logits_g = self.head_global(g)                   # (B,5)

        tokens = z[:, 1:]                                # (B,N,D)
        tokens = tokens.transpose(1, 2).contiguous()     # (B,D,N)
        tokens = tokens.view(B, -1, self.grid, self.grid)  # (B,D,g,g)

        logits_low = self.head_map(tokens)               # (B,5,g,g)
        logits_m = F.interpolate(logits_low, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return logits_g, logits_m


def load_diagnoser_map(ckpt_path: str, img_size: int, vit_patch: int, vit_dim: int, vit_depth: int, vit_heads: int, device: torch.device):
    model = TinyViTDiagnoserMap(
        img_size=img_size,
        patch_size=vit_patch,
        embed_dim=vit_dim,
        depth=vit_depth,
        num_heads=vit_heads,
        num_labels=5,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # backward compat: head.* -> head_global.*
    if "head.weight" in sd and "head_global.weight" not in sd:
        sd2 = dict(sd)
        sd2["head_global.weight"] = sd2.pop("head.weight")
        sd2["head_global.bias"] = sd2.pop("head.bias")
        sd = sd2

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, missing, unexpected


# ============================================================
# ToolBank helpers (LoRA load)
# ============================================================
def lora_state_dict_from_ckpt(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "lora_state_dict" in ckpt:
        return ckpt["lora_state_dict"]
    # fallback: some saved as plain sd
    if all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise KeyError("Could not find lora_state_dict in ckpt.")


def load_lora_for_action(tb: ToolBank, action: str, sd: Dict[str, torch.Tensor], strict: bool = True):
    if hasattr(tb, "load_lora_state_dict_for_action"):
        return tb.load_lora_state_dict_for_action(action, sd, strict=strict)
    if hasattr(tb, "load_lora_state_dict"):
        return tb.load_lora_state_dict(action, sd, strict=strict)
    # last-resort
    missing, unexpected = tb.load_state_dict(sd, strict=False)
    if strict and len(unexpected) > 0:
        raise RuntimeError(f"Unexpected keys while loading LoRA for {action}: {unexpected[:10]}")
    return {"missing": missing, "unexpected": unexpected}


# ============================================================
# Build eval set from lora_dataset pairs
# ============================================================
def build_unique_eval_pairs(data_root: str, splits: List[str], actions: List[str], max_items: int, seed: int) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Returns unique list of (inp_path, gt_path, meta).
    We gather pairs from all actions, then de-duplicate by inp_path.
    """
    import random
    rng = random.Random(seed)

    uniq: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
    for sp in splits:
        for action in actions:
            cfg = ActionPairConfig(action=action, split=sp, data_root=data_root)
            pairs, reports = build_action_pairs(cfg)
            for inp_path, gt_path, meta in pairs:
                if inp_path not in uniq:
                    m = dict(meta) if isinstance(meta, dict) else {"meta": meta}
                    m["split"] = sp
                    m["source_action"] = action
                    uniq[inp_path] = (inp_path, gt_path, m)

    items = list(uniq.values())
    rng.shuffle(items)
    if max_items > 0:
        items = items[:max_items]
    return items


# ============================================================
# Main
# ============================================================
@dataclass
class Cfg:
    data_root: str
    splits: List[str]
    out_json: str

    diagnoser_ckpt: str
    diagnoser_img_size: int
    vit_patch: int
    vit_dim: int
    vit_depth: int
    vit_heads: int

    toolbank_lora_root: str
    backbone_ckpt: str
    dim: int
    bias: int
    volterra_rank: int
    lora_rank: int
    lora_alpha: float
    wrap_only_1x1: int

    scales: List[float]
    actions: List[str]

    max_items: int
    save_every: int
    seed: int

    use_amp: int
    channels_last: int
    tf32: int

    save_preview: int
    preview_dir: str


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", default="E:/ReAct-IR/data")
    ap.add_argument("--splits", nargs="+", default=["train"], help="train/test (your lora_dataset supports train|test)")
    ap.add_argument("--out_json", default="E:/ReAct-IR/rollouts/rollouts_train.json")

    ap.add_argument("--diagnoser_ckpt", required=True)
    ap.add_argument("--diagnoser_img_size", type=int, default=256)
    ap.add_argument("--vit_patch", type=int, default=16)
    ap.add_argument("--vit_dim", type=int, default=384)
    ap.add_argument("--vit_depth", type=int, default=6)
    ap.add_argument("--vit_heads", type=int, default=6)

    ap.add_argument("--toolbank_lora_root", required=True)
    ap.add_argument("--backbone_ckpt", default="E:/ReAct-IR/checkpoints/backbone/epoch_019_L0.0230_P30.61_S0.9292.pth")
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--volterra_rank", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=2)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--wrap_only_1x1", type=int, default=1)

    ap.add_argument("--scales", nargs="+", default=["0", "0.25", "0.5", "0.75", "1.0"])
    ap.add_argument("--actions", nargs="+", default=ACTIONS_INTERNAL)

    ap.add_argument("--max_items", type=int, default=-1, help="-1 for all")
    ap.add_argument("--save_every", type=int, default=50, help="flush json every N samples (jsonl always flushes)")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=1)

    ap.add_argument("--save_preview", type=int, default=0, help="1 to save (input|pred|gt) previews occasionally")
    ap.add_argument("--preview_dir", default="E:/ReAct-IR/rollouts/previews")

    a = ap.parse_args()

    scales = parse_scales(a.scales)
    return Cfg(
        data_root=a.data_root,
        splits=[s.strip() for s in a.splits],
        out_json=a.out_json,
        diagnoser_ckpt=a.diagnoser_ckpt,
        diagnoser_img_size=int(a.diagnoser_img_size),
        vit_patch=int(a.vit_patch),
        vit_dim=int(a.vit_dim),
        vit_depth=int(a.vit_depth),
        vit_heads=int(a.vit_heads),
        toolbank_lora_root=a.toolbank_lora_root,
        backbone_ckpt=a.backbone_ckpt,
        dim=int(a.dim),
        bias=int(a.bias),
        volterra_rank=int(a.volterra_rank),
        lora_rank=int(a.lora_rank),
        lora_alpha=float(a.lora_alpha),
        wrap_only_1x1=int(a.wrap_only_1x1),
        scales=scales,
        actions=[s.strip() for s in a.actions],
        max_items=int(a.max_items),
        save_every=int(a.save_every),
        seed=int(a.seed),
        use_amp=int(a.use_amp),
        channels_last=int(a.channels_last),
        tf32=int(a.tf32),
        save_preview=int(a.save_preview),
        preview_dir=a.preview_dir,
    )


def save_triplet_preview(inp: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, out_path: str):
    """
    inp/pred/gt: (1,3,H,W)
    """
    inp_u8 = (inp[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    pr_u8 = (pred[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    gt_u8 = (gt[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)

    h, w = inp_u8.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = inp_u8
    canvas[:, w:2*w] = pr_u8
    canvas[:, 2*w:3*w] = gt_u8

    safe_makedirs(os.path.dirname(out_path))
    Image.fromarray(canvas).save(out_path)


def main():
    cfg = parse_args()
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[SKIMAGE SSIM]", "ON" if USE_SKIMAGE else "OFF")

    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    safe_makedirs(os.path.dirname(cfg.out_json))
    if cfg.save_preview:
        safe_makedirs(cfg.preview_dir)

    # ---------------- load diagnoser map ----------------
    diagnoser, missing, unexpected = load_diagnoser_map(
        cfg.diagnoser_ckpt,
        img_size=cfg.diagnoser_img_size,
        vit_patch=cfg.vit_patch,
        vit_dim=cfg.vit_dim,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        device=device,
    )
    print("[Diagnoser]", cfg.diagnoser_ckpt)
    print(f"[Diagnoser] loaded strict=False missing={len(missing)} unexpected={len(unexpected)}")

    # ---------------- build eval list ----------------
    pairs = build_unique_eval_pairs(
        data_root=cfg.data_root,
        splits=cfg.splits,
        actions=cfg.actions,
        max_items=cfg.max_items,
        seed=cfg.seed,
    )
    print(f"[Pairs] total unique inputs = {len(pairs)}  splits={cfg.splits}")

    # ---------------- load backbone + toolbank ----------------
    base = VETNet(dim=cfg.dim, bias=bool(cfg.bias), volterra_rank=cfg.volterra_rank)
    ckpt = torch.load(cfg.backbone_ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing_b, unexpected_b = base.load_state_dict(sd, strict=False)
    for p in base.parameters():
        p.requires_grad = False

    tb = ToolBank(
        base,
        actions=cfg.actions + ["A_STOP"],
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        wrap_only_1x1=bool(cfg.wrap_only_1x1),
    ).to(device)

    if cfg.channels_last and device.type == "cuda":
        tb = tb.to(memory_format=torch.channels_last)
        base = base.to(memory_format=torch.channels_last)

    print("[Backbone] ckpt =", cfg.backbone_ckpt)
    print(f"[Backbone] loaded strict=False missing={len(missing_b)} unexpected={len(unexpected_b)}")
    if len(missing_b) > 0:
        print("  example missing:", missing_b[:10])
    if len(unexpected_b) > 0:
        print("  example unexpected:", unexpected_b[:10])

    # ---------------- load all loras into toolbank ----------------
    lora_paths: Dict[str, str] = {}
    for a in cfg.actions:
        alias = None
        # map internal -> folder name
        if a == "A_DEBLUR": alias = "deblur"
        if a == "A_DERAIN": alias = "derain"
        if a == "A_DESNOW": alias = "desnow"
        if a == "A_DEHAZE": alias = "dehaze"
        if a == "A_DEDROP": alias = "dedrop"
        if alias is None:
            raise ValueError(f"Unknown action {a}")

        action_dir = os.path.join(cfg.toolbank_lora_root, alias)
        ckpt_path = choose_best_lora_ckpt(action_dir)
        lora_paths[a] = ckpt_path

        ck = torch.load(ckpt_path, map_location="cpu")
        lsd = lora_state_dict_from_ckpt(ck)
        load_lora_for_action(tb, a, lsd, strict=False)

    print("[LoRA] loaded:")
    for a, p in lora_paths.items():
        print(f"  - {a}: {p}")

    # ---------------- output format ----------------
    is_jsonl = cfg.out_json.lower().endswith(".jsonl")
    out_f = None
    results: List[Dict[str, Any]] = []

    if is_jsonl:
        out_f = open(cfg.out_json, "w", encoding="utf-8")

    # ---------------- rollout loop ----------------
    tb.eval()
    diagnoser.eval()

    t_start = time.time()
    for idx, (inp_path, gt_path, meta) in enumerate(pairs, start=1):
        try:
            img_in = read_rgb(inp_path)
            img_gt = read_rgb(gt_path)
        except Exception as e:
            print(f"[WARN] skip (read error) {inp_path} : {e}")
            continue

        # NOTE: For diagnoser map, we resize to 256x256 (consistent with training)
        img_in_d = img_in.resize((cfg.diagnoser_img_size, cfg.diagnoser_img_size), Image.BILINEAR)
        x_d = to_tensor_chw(img_in_d).unsqueeze(0).to(device)

        # For restoration, use original size with pad-to-8
        x = to_tensor_chw(img_in).unsqueeze(0).to(device)
        y = to_tensor_chw(img_gt).unsqueeze(0).to(device)

        if cfg.channels_last and device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)
            x_d = x_d.to(memory_format=torch.channels_last)

        x, pad = pad_to_multiple(x, mul=8)
        y, pad2 = pad_to_multiple(y, mul=8)
        # if pads differ due to mismatch sizes (rare), just trust separate pads
        # we'll unpad using each pad

        use_amp = (cfg.use_amp == 1 and device.type == "cuda")

        # --- diagnoser state ---
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lg, lm = diagnoser(x_d)
            s0 = torch.sigmoid(lg[0]).float().detach().cpu()     # (5,)
            m0 = torch.sigmoid(lm[0]).float().detach().cpu()     # (5,256,256)

        # compress map to lightweight stats (store full map would explode json)
        # store mean intensity per channel + top-k hotspots (optional)
        map_mean = m0.view(5, -1).mean(dim=1).tolist()
        map_max = m0.view(5, -1).amax(dim=1).tolist()

        # baseline metrics: input vs gt
        with torch.no_grad():
            x0 = unpad(x, pad).clamp(0, 1)
            y0 = unpad(y, pad2).clamp(0, 1)

        psnr_in = psnr_torch(x0, y0)
        ssim_in = ssim_optional(x0, y0) if USE_SKIMAGE else 0.0

        # evaluate all actions/scales
        action_logs = []
        for a in cfg.actions:
            for sc in cfg.scales:
                with torch.no_grad():
                    tb.activate(a, scale=float(sc))
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        pred = tb(x)  # (1,3,H',W')
                    pred = pred.detach().clamp(0, 1)
                    pred_u = unpad(pred, pad).clamp(0, 1)

                ps = psnr_torch(pred_u, y0)
                ss = ssim_optional(pred_u, y0) if USE_SKIMAGE else 0.0

                action_logs.append({
                    "action": a,
                    "scale": float(sc),
                    "psnr": float(ps),
                    "ssim": float(ss),
                    "d_psnr": float(ps - psnr_in),
                    "d_ssim": float(ss - ssim_in) if USE_SKIMAGE else 0.0,
                })

        # choose best by d_psnr (tie -> higher psnr)
        best = max(action_logs, key=lambda r: (r["d_psnr"], r["psnr"]))
        pred_action = best["action"]
        pred_scale = best["scale"]

        # optional preview save
        if cfg.save_preview and (idx % max(cfg.save_every, 1) == 0):
            # re-run best once for preview
            with torch.no_grad():
                tb.activate(pred_action, scale=float(pred_scale))
                with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred = tb(x).detach().clamp(0, 1)
                pred_u = unpad(pred, pad).clamp(0, 1)
            outp = os.path.join(cfg.preview_dir, f"{idx:06d}_{pred_action}_s{pred_scale:.2f}.png")
            save_triplet_preview(x0, pred_u, y0, outp)

        rec = {
            "id": int(idx),
            "input": inp_path.replace("\\", "/"),
            "gt": gt_path.replace("\\", "/"),
            "meta": meta,
            "state": {
                "s0": [float(v) for v in s0.tolist()],           # 5
                "m0_mean": [float(v) for v in map_mean],         # 5
                "m0_max": [float(v) for v in map_max],           # 5
            },
            "baseline": {
                "psnr_in": float(psnr_in),
                "ssim_in": float(ssim_in),
            },
            "sweep": action_logs,
            "best": best,  # best action/scale by d_psnr
        }

        if is_jsonl:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
        else:
            results.append(rec)
            if cfg.save_every > 0 and (idx % cfg.save_every == 0):
                with open(cfg.out_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        # progress log
        if idx % 20 == 0:
            elapsed = time.time() - t_start
            it_s = idx / max(elapsed, 1e-6)
            eta = (len(pairs) - idx) / max(it_s, 1e-6)
            print(f"[{idx:6d}/{len(pairs)}] psnr_in={psnr_in:.2f} best=({pred_action}, {pred_scale}) dP={best['d_psnr']:+.2f} ETA={eta/60:.1f}m")

    if is_jsonl:
        out_f.close()
    else:
        with open(cfg.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("[Done] wrote:", cfg.out_json)
    print("[Note] JSONL recommended for large rollouts (append-safe).")


if __name__ == "__main__":
    main()
