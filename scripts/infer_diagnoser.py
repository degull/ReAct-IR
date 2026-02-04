# E:/ReAct-IR/scripts/infer_diagnoser.py
# ------------------------------------------------------------
# Inference for Map Diagnoser (vit_map / cnn_map)
# - strict load guaranteed (same model definitions as train_diagnoser.py)
# - supports checkpoints that store 'state_dict' or 'model_state_dict'
# - prints: s0 (sigmoid probs), pred_action, and m0_mean/m0_max from map
# ------------------------------------------------------------
""" import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.amp import autocast

# label order must match train_diagnoser.py
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
IDX_TO_ACTION = {
    0: "A_DEBLUR",
    1: "A_DERAIN",
    2: "A_DESNOW",
    3: "A_DEHAZE",
    4: "A_DEDROP",
}


# ============================================================
# Models (EXACT SAME as train_diagnoser.py)
# ============================================================
class ViTMapDiagnoser(nn.Module):
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
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
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

        self.map_head = nn.Linear(embed_dim, num_labels)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.map_head.weight, std=0.02)
        nn.init.zeros_(self.map_head.bias)

    def forward(self, x: torch.Tensor):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)                  # (B,D,Hp,Wp)
        x = x.flatten(2).transpose(1, 2)         # (B,N,D)

        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)

        x = self.encoder(x)                      # (B,N,D)
        x = self.norm(x)

        patch_logits = self.map_head(x)          # (B,N,5)
        m0 = patch_logits.transpose(1, 2).reshape(B, 5, self.grid, self.grid)  # (B,5,Hp,Wp)

        logit_mean = m0.mean(dim=(2, 3))
        logit_max = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)  # (B,5)

        return logits, m0


class CNNMapDiagnoser(nn.Module):
    def __init__(self, num_labels: int = 5, width: int = 64):
        super().__init__()
        w = width
        self.feat = nn.Sequential(
            nn.Conv2d(3, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True),          # /2
            nn.Conv2d(w, 2*w, 3, 2, 1), nn.ReLU(inplace=True),        # /4
            nn.Conv2d(2*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /8
            nn.Conv2d(4*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /16
        )
        self.map_head = nn.Conv2d(4*w, num_labels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        f = self.feat(x)
        m0 = self.map_head(f)  # (B,5,h,w)

        logit_mean = m0.mean(dim=(2, 3))
        logit_max = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)

        return logits, m0


# ============================================================
# Helpers
# ============================================================
def load_img_tensor(path: str, patch: int = 256, crop: str = "center") -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    W, H = img.size

    if crop == "center":
        if H >= patch and W >= patch:
            x0 = (W - patch) // 2
            y0 = (H - patch) // 2
            img = img.crop((x0, y0, x0 + patch, y0 + patch))
        else:
            img = img.resize((patch, patch), resample=Image.BILINEAR)
    else:
        img = img.resize((patch, patch), resample=Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


def _pick_state_dict(ckpt: dict) -> dict:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    return ckpt  # fallback


def build_model_from_ckpt_cfg(ckpt: dict, override_model: str = "") -> nn.Module:
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    model_name = override_model.strip() or str(cfg.get("model", "vit_map"))

    patch = int(cfg.get("patch", 256))
    if model_name == "vit_map":
        vit_patch = int(cfg.get("vit_patch", 16))
        vit_dim = int(cfg.get("vit_dim", 384))
        vit_depth = int(cfg.get("vit_depth", 6))
        vit_heads = int(cfg.get("vit_heads", 6))
        return ViTMapDiagnoser(
            img_size=patch,
            patch_size=vit_patch,
            embed_dim=vit_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            num_labels=5,
        )
    if model_name == "cnn_map":
        cnn_width = int(cfg.get("cnn_width", 64))
        return CNNMapDiagnoser(num_labels=5, width=cnn_width)

    raise ValueError(f"Unknown model_name='{model_name}'. Expected vit_map or cnn_map.")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth")
    ap.add_argument("--input", required=True)
    ap.add_argument("--use_amp", type=int, default=1)

    ap.add_argument("--model", type=str, default="", choices=["", "vit_map", "cnn_map"])
    ap.add_argument("--patch", type=int, default=-1, help="override patch size; -1 uses ckpt cfg.patch")
    ap.add_argument("--crop", type=str, default="center", choices=["center", "resize"])

    ap.add_argument("--print_logits", type=int, default=1)
    ap.add_argument("--print_map_stats", type=int, default=1)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format unexpected: expected dict with cfg/state_dict keys.")

    cfg = ckpt.get("cfg", {})
    cfg_patch = int(cfg.get("patch", 256))
    patch = int(args.patch) if int(args.patch) > 0 else cfg_patch

    model = build_model_from_ckpt_cfg(ckpt, override_model=args.model)
    sd = _pick_state_dict(ckpt)

    # strict load guarantee
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    x = load_img_tensor(args.input, patch=patch, crop=args.crop).to(device)

    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.float16, enabled=(bool(args.use_amp) and device.type == "cuda")):
            logits, m0 = model(x)

        probs = torch.sigmoid(logits)[0].detach().float().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_action = IDX_TO_ACTION[pred_idx]

    print("[CKPT]", args.ckpt)
    print("[Model]", ckpt.get("model", cfg.get("model", "unknown")))
    print("[Patch]", patch, "| crop=", args.crop)
    print("[Input]", args.input)

    if args.print_logits:
        s0 = {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)}
        print("[s0] " + "  ".join([f"{k}={s0[k]:.3f}" for k in LABEL_NAMES]))
        print("[pred_action]", pred_action)

    if args.print_map_stats:
        m0_mean = m0.mean(dim=(2, 3))[0].detach().float().cpu().numpy()
        m0_max = m0.amax(dim=(2, 3))[0].detach().float().cpu().numpy()
        print("[m0_mean(logit)] " + "  ".join([f"{LABEL_NAMES[i]}={m0_mean[i]:.3f}" for i in range(5)]))
        print("[m0_max (logit)] " + "  ".join([f"{LABEL_NAMES[i]}={m0_max[i]:.3f}" for i in range(5)]))

        m0_sig = torch.sigmoid(m0)
        m0p_mean = m0_sig.mean(dim=(2, 3))[0].detach().float().cpu().numpy()
        m0p_max = m0_sig.amax(dim=(2, 3))[0].detach().float().cpu().numpy()
        print("[m0_mean(sig)]   " + "  ".join([f"{LABEL_NAMES[i]}={m0p_mean[i]:.3f}" for i in range(5)]))
        print("[m0_max (sig)]   " + "  ".join([f"{LABEL_NAMES[i]}={m0p_max[i]:.3f}" for i in range(5)]))


if __name__ == "__main__":
    main()
 """


# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/CSD/Test/Snow/1.tif"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/CSD/Test/Snow/2.tif"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/CSD/Test/Snow/3.tif"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/CSD/Test/Snow/4.tif"

# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/DayRainDrop/Drop/00001/00001.png"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/DayRainDrop/Blur/00001/00001.png"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/rain100H/test/rain/norain-1.png"
# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "E:/ReAct-IR/data/RESIDE-6K/test/hazy/0001_0.8_0.2.jpg"


# python scripts/infer_diagnoser.py --ckpt "E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth" --input "..." --model vit_map

# scripts/infer_diagnoser.py
# ------------------------------------------------------------
# Robust Diagnoser Inference (auto ckpt format)
# - Supports:
#   (A) "diagnoser_best.pth" style: pos_embed=[1,256,D], map_head=Linear, no cls_token
#   (B) "diagnoser_1/diagnoser_map_best.pth" style: pos_embed=[1,257,D], has cls_token,
#       head_map (Conv 1x1) and optional head_global
#
# Features:
# - Auto-detect vit_map / cnn_map from ckpt["model"] or ckpt["cfg"]["model"]
# - Auto-adapt state_dict keys and shapes:
#     * drop cls_token if target doesn't have it
#     * rename head_map.* <-> map_head.* depending on module type
#     * adapt head_map.weight  [5,384,1,1]  <->  [5,384]  (squeeze/unsqueeze)
#     * adapt pos_embed length 257 <-> 256 by dropping/adding CLS token position
# - Optional label permutation: --diag_perm 3 2 1 4 0
# - Prints:
#     s0 (sigmoid), predicted action (argmax), m0_mean/m0_max (logit & sigmoid)
# ------------------------------------------------------------

import os
import sys
import argparse
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# -------------------------
# Project import-safe
# -------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ACTIONS = ["blur", "rain", "snow", "haze", "drop"]
ACTIONS_AGENT = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dprint(debug: int, *args, **kwargs):
    if int(debug) == 1:
        print(*args, **kwargs)


# -------------------------
# CKPT helpers
# -------------------------
def load_ckpt_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                # Heuristic: "model" sometimes is nested or metadata; but if it's tensor dict, return it.
                if all(isinstance(v, torch.Tensor) for v in ckpt[k].values()):
                    return ckpt[k]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise RuntimeError(f"Cannot extract state_dict from ckpt: {path}")


def read_ckpt_meta(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {}
    meta = {}
    if "model" in ckpt and not isinstance(ckpt["model"], dict):
        meta["model"] = ckpt["model"]
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        meta["cfg"] = ckpt["cfg"]
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        meta["args"] = ckpt["args"]
    return meta


def infer_model_type_from_ckpt(ckpt_path: str, forced: Optional[str] = None) -> str:
    if forced is not None:
        forced = str(forced).strip().lower()
        if forced in ["vit_map", "cnn_map"]:
            return forced
        raise ValueError(f"--model_type must be vit_map or cnn_map, got {forced}")

    meta = read_ckpt_meta(ckpt_path)
    model_type = None

    if "model" in meta:
        model_type = meta["model"]
    if model_type is None and "cfg" in meta:
        model_type = meta["cfg"].get("model", None)

    if model_type is None:
        # fallback: look at state_dict keys
        sd = load_ckpt_state(ckpt_path)
        keys = set(sd.keys())
        if any("encoder.layers" in k for k in keys) or any("pos_embed" in k for k in keys):
            model_type = "vit_map"
        else:
            model_type = "cnn_map"

    model_type = str(model_type).strip().lower()
    if model_type not in ["vit_map", "cnn_map"]:
        raise RuntimeError(f"Unknown model_type in ckpt: {model_type}")
    return model_type


# -------------------------
# Models (2 variants)
# -------------------------
class ViTMapDiagnoser(nn.Module):
    """
    Target architecture for inference:
    - patch_embed: Conv stride=patch
    - pos_embed: [1, N, D]  (NO CLS by default)
    - encoder: TransformerEncoder
    - map_head: Linear(D->5) producing per-patch logits
    - output: (logits_global[B,5], m0[B,5,grid,grid])
    """
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
        use_cls_token: bool = False,  # allow building CLS variant too
        map_head_type: str = "linear",  # "linear" or "conv"
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid = self.img_size // self.patch_size
        self.num_patches = self.grid * self.grid
        self.use_cls_token = bool(use_cls_token)

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        # pos_embed length depends on cls usage
        pos_len = self.num_patches + (1 if self.use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

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

        map_head_type = str(map_head_type).lower().strip()
        assert map_head_type in ["linear", "conv"]
        self.map_head_type = map_head_type
        if self.map_head_type == "linear":
            self.map_head = nn.Linear(embed_dim, num_labels)
        else:
            self.map_head = nn.Conv2d(embed_dim, num_labels, kernel_size=1, stride=1, padding=0)

        # optional global head (some ckpts have it)
        self.head_global = nn.Linear(embed_dim, num_labels)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        # init heads
        if self.map_head_type == "linear":
            nn.init.trunc_normal_(self.map_head.weight, std=0.02)
            nn.init.zeros_(self.map_head.bias)
        else:
            nn.init.trunc_normal_(self.map_head.weight, std=0.02)
            if self.map_head.bias is not None:
                nn.init.zeros_(self.map_head.bias)
        nn.init.trunc_normal_(self.head_global.weight, std=0.02)
        nn.init.zeros_(self.head_global.bias)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.patch_embed(x)                  # (B,D,Hp,Wp)
        x = x.flatten(2).transpose(1, 2)         # (B,N,D)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)   # (B,1,D)
            x = torch.cat([cls, x], dim=1)           # (B,1+N,D)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x)

        if self.use_cls_token:
            x_tokens = x[:, 1:, :]  # patches
            x_cls = x[:, 0, :]
        else:
            x_tokens = x
            x_cls = x.mean(dim=1)

        if self.map_head_type == "linear":
            patch_logits = self.map_head(x_tokens)  # (B,N,5)
            m0 = patch_logits.transpose(1, 2).reshape(B, 5, self.grid, self.grid)
        else:
            # conv head expects (B,D,Hp,Wp)
            f = x_tokens.transpose(1, 2).reshape(B, x_tokens.shape[-1], self.grid, self.grid)
            m0 = self.map_head(f)  # (B,5,grid,grid)

        # global logits: combine mean/max on map (consistent with your run_agent)
        logit_mean = m0.mean(dim=(2, 3))
        logit_max = m0.amax(dim=(2, 3))
        logits_map = 0.5 * (logit_mean + logit_max)

        # also support explicit global head (some ckpts)
        logits_global = self.head_global(x_cls)

        # final logits: average of both when cls head exists (robust)
        logits = 0.5 * (logits_map + logits_global)
        return logits, m0


class CNNMapDiagnoser(nn.Module):
    def __init__(self, num_labels: int = 5, width: int = 64):
        super().__init__()
        w = int(width)
        self.img_size = 256
        self.feat = nn.Sequential(
            nn.Conv2d(3, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True),          # /2
            nn.Conv2d(w, 2*w, 3, 2, 1), nn.ReLU(inplace=True),        # /4
            nn.Conv2d(2*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /8
            nn.Conv2d(4*w, 4*w, 3, 2, 1), nn.ReLU(inplace=True),      # /16
        )
        self.map_head = nn.Conv2d(4*w, num_labels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        f = self.feat(x)
        m0 = self.map_head(f)  # (B,5,h,w)
        logit_mean = m0.mean(dim=(2, 3))
        logit_max  = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)
        return logits, m0


# -------------------------
# Preprocess
# -------------------------
def load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t.to(DEVICE)


def preprocess_for_diagnoser(x: torch.Tensor, img_size: int, mode: str) -> torch.Tensor:
    _, _, H, W = x.shape
    if H == img_size and W == img_size:
        return x
    mode = str(mode).lower().strip()
    if mode == "resize":
        return F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    if mode == "center_crop":
        if H >= img_size and W >= img_size:
            top = (H - img_size) // 2
            left = (W - img_size) // 2
            return x[:, :, top:top+img_size, left:left+img_size]
        return F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# State dict adaptation (the core)
# -------------------------
def _has_key(sd: Dict[str, torch.Tensor], k: str) -> bool:
    return (k in sd) and torch.is_tensor(sd[k])


def _rename_prefix(sd: Dict[str, torch.Tensor], src: str, dst: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith(src):
            out[dst + k[len(src):]] = v
        else:
            out[k] = v
    return out


def adapt_vit_state_dict_to_model(sd: Dict[str, torch.Tensor], model: ViTMapDiagnoser, debug: int = 0) -> Dict[str, torch.Tensor]:
    """
    Make checkpoint sd compatible with our ViTMapDiagnoser instance.
    Handles:
      - cls_token present in ckpt but not in model (drop)
      - pos_embed length mismatch (257 vs 256)
      - head_map <-> map_head naming and shape (conv vs linear)
      - optional head_global
    """
    sd2 = dict(sd)

    # 0) Some ckpts store "head_map" / "head_global" naming
    # Our model uses: "map_head" and "head_global"
    if any(k.startswith("head_map.") for k in sd2.keys()) and not any(k.startswith("map_head.") for k in sd2.keys()):
        sd2 = _rename_prefix(sd2, "head_map.", "map_head.")
    if any(k.startswith("head_global.") for k in sd2.keys()) and not any(k.startswith("head_global.") for k in sd2.keys()):
        # (no-op) kept for symmetry
        pass

    # 1) Drop cls_token if model doesn't have it
    if (model.cls_token is None) and _has_key(sd2, "cls_token"):
        dprint(debug, "[ADAPT] drop cls_token (model has no cls_token)")
        sd2.pop("cls_token", None)

    # 2) pos_embed length adapt
    if _has_key(sd2, "pos_embed") and hasattr(model, "pos_embed"):
        ck = sd2["pos_embed"]
        mk = model.pos_embed
        if ck.shape != mk.shape:
            # both are [1, L, D]
            if ck.dim() == 3 and mk.dim() == 3 and ck.shape[0] == mk.shape[0] and ck.shape[2] == mk.shape[2]:
                Lc, Lm = ck.shape[1], mk.shape[1]
                if Lc == Lm + 1:
                    # ckpt has CLS pos, model doesn't -> drop first token
                    dprint(debug, f"[ADAPT] pos_embed {Lc}->{Lm}: drop first (CLS) position")
                    sd2["pos_embed"] = ck[:, 1:, :].contiguous()
                elif Lc + 1 == Lm:
                    # ckpt has no CLS pos, model has -> pad by duplicating first token
                    dprint(debug, f"[ADAPT] pos_embed {Lc}->{Lm}: add CLS position (duplicate first)")
                    cls_pos = ck[:, :1, :].contiguous()
                    sd2["pos_embed"] = torch.cat([cls_pos, ck], dim=1).contiguous()
                else:
                    # generic: truncate or pad zeros
                    dprint(debug, f"[ADAPT] pos_embed {Lc}->{Lm}: generic truncate/pad")
                    if Lc > Lm:
                        sd2["pos_embed"] = ck[:, :Lm, :].contiguous()
                    else:
                        pad = torch.zeros((1, Lm - Lc, ck.shape[2]), dtype=ck.dtype)
                        sd2["pos_embed"] = torch.cat([ck, pad], dim=1).contiguous()

    # 3) map_head weight shape adapt between conv and linear
    # model.map_head can be Linear: weight [5,D], or Conv2d: weight [5,D,1,1]
    if hasattr(model, "map_head"):
        if isinstance(model.map_head, nn.Linear):
            # expect [5, D]
            if _has_key(sd2, "map_head.weight") and sd2["map_head.weight"].dim() == 4:
                w = sd2["map_head.weight"]
                if w.shape[2:] == (1, 1):
                    dprint(debug, "[ADAPT] map_head.weight conv->linear squeeze")
                    sd2["map_head.weight"] = w[:, :, 0, 0].contiguous()
            if _has_key(sd2, "map_head.bias") and sd2["map_head.bias"].dim() != 1:
                sd2["map_head.bias"] = sd2["map_head.bias"].reshape(-1).contiguous()

        elif isinstance(model.map_head, nn.Conv2d):
            # expect [5, D,1,1]
            if _has_key(sd2, "map_head.weight") and sd2["map_head.weight"].dim() == 2:
                w = sd2["map_head.weight"]
                dprint(debug, "[ADAPT] map_head.weight linear->conv unsqueeze")
                sd2["map_head.weight"] = w[:, :, None, None].contiguous()
            # bias should be [5]
            if _has_key(sd2, "map_head.bias") and sd2["map_head.bias"].dim() != 1:
                sd2["map_head.bias"] = sd2["map_head.bias"].reshape(-1).contiguous()

    # 4) If ckpt has "head.*" instead of "head_global.*"
    # (Some variants might name global head as "head" or "head_global")
    if any(k.startswith("head.") for k in sd2.keys()) and not any(k.startswith("head_global.") for k in sd2.keys()):
        # rename "head." -> "head_global."
        sd2 = _rename_prefix(sd2, "head.", "head_global.")

    # 5) If model doesn't have head_global (it does), ignore anyway via strict=False later.
    return sd2


def choose_vit_variant_from_ckpt(sd: Dict[str, torch.Tensor]) -> Tuple[bool, str]:
    """
    Decide whether to build model with cls_token and whether map_head is linear/conv.
    """
    use_cls = _has_key(sd, "cls_token") or (_has_key(sd, "pos_embed") and sd["pos_embed"].shape[1] == 257)

    # If ckpt map head looks like conv weight -> use conv
    map_head_type = "linear"
    if _has_key(sd, "head_map.weight"):
        w = sd["head_map.weight"]
        if w.dim() == 4:
            map_head_type = "conv"
    if _has_key(sd, "map_head.weight"):
        w = sd["map_head.weight"]
        if w.dim() == 4:
            map_head_type = "conv"

    return use_cls, map_head_type


def build_vit_from_ckpt_cfg(meta: Dict[str, Any], sd: Dict[str, torch.Tensor], debug: int = 0) -> ViTMapDiagnoser:
    cfg = {}
    if isinstance(meta.get("cfg", None), dict):
        cfg = meta["cfg"]

    # Your ckpt sometimes used keys like patch/vit_patch/vit_dim...
    # We'll infer robustly with fallbacks.
    img_size = int(cfg.get("img_size", cfg.get("patch", 256)))
    patch_size = int(cfg.get("patch_size", cfg.get("vit_patch", 16)))
    embed_dim = int(cfg.get("embed_dim", cfg.get("vit_dim", 384)))
    depth = int(cfg.get("depth", cfg.get("vit_depth", 6)))
    num_heads = int(cfg.get("num_heads", cfg.get("vit_heads", 6)))
    mlp_ratio = float(cfg.get("mlp_ratio", 4.0))
    dropout = float(cfg.get("dropout", 0.0))

    use_cls, map_head_type = choose_vit_variant_from_ckpt(sd)

    # If ckpt pos_embed length is 256 => usually no CLS, but we can still set use_cls False
    if _has_key(sd, "pos_embed"):
        L = int(sd["pos_embed"].shape[1])
        if L == (img_size // patch_size) ** 2:
            use_cls = False
        elif L == (img_size // patch_size) ** 2 + 1:
            use_cls = True

    model = ViTMapDiagnoser(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_labels=5,
        use_cls_token=use_cls,
        map_head_type=map_head_type,
    )

    dprint(debug, "[DEBUG][Diagnoser] vit_map inferred cfg:", {
        "img_size": img_size, "patch_size": patch_size, "in_chans": 3,
        "embed_dim": embed_dim, "depth": depth, "num_heads": num_heads,
        "mlp_ratio": mlp_ratio, "dropout": dropout, "num_labels": 5,
        "use_cls_token": use_cls, "map_head_type": map_head_type
    })
    return model


def load_diagnoser(ckpt_path: str, model_type: str, debug: int = 0) -> nn.Module:
    meta = read_ckpt_meta(ckpt_path)
    sd = load_ckpt_state(ckpt_path)

    if model_type == "cnn_map":
        # cfg width if exists
        cfg = meta.get("cfg", {}) if isinstance(meta.get("cfg", {}), dict) else {}
        width = int(cfg.get("cnn_width", 64))
        model = CNNMapDiagnoser(num_labels=5, width=width)
        # load best-effort
        msg = model.load_state_dict(sd, strict=False)
        dprint(debug, f"[DEBUG][Diagnoser] load cnn_map strict=False | missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        model.to(DEVICE).eval()
        return model

    # vit_map
    model = build_vit_from_ckpt_cfg(meta, sd, debug=debug)
    sd2 = adapt_vit_state_dict_to_model(sd, model, debug=debug)

    msg = model.load_state_dict(sd2, strict=False)
    dprint(debug, f"[DEBUG][Diagnoser] load vit_map strict=False | missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if int(debug) == 1 and (len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0):
        # print a few keys to help
        dprint(1, "[DEBUG][Diagnoser] missing (first 20):", msg.missing_keys[:20])
        dprint(1, "[DEBUG][Diagnoser] unexpected (first 20):", msg.unexpected_keys[:20])

    model.to(DEVICE).eval()
    return model


# -------------------------
# Permutation helper
# -------------------------
def apply_perm_5(x: torch.Tensor, perm: Optional[List[int]]) -> torch.Tensor:
    if perm is None:
        return x
    if len(perm) != 5:
        raise ValueError(f"--diag_perm must have 5 ints, got {perm}")
    if sorted(perm) != [0, 1, 2, 3, 4]:
        raise ValueError(f"--diag_perm must be a permutation of 0..4, got {perm}")
    # x: [B,5] or [B,5,H,W]
    if x.dim() == 2:
        return x[:, perm]
    if x.dim() == 4:
        return x[:, perm, :, :]
    raise ValueError(f"apply_perm_5 expects 2D or 4D tensor, got {x.shape}")


# -------------------------
# Main inference
# -------------------------
@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)

    p.add_argument("--model_type", default=None, choices=["vit_map", "cnn_map"],
                   help="Force model type. If omitted, auto-detect from ckpt.")

    p.add_argument("--diag_mode", default="center_crop", choices=["resize", "center_crop"])
    p.add_argument("--diag_img_size", type=int, default=256)

    p.add_argument("--diag_perm", nargs="+", type=int, default=None,
                   help="Optional label permutation for 5 dims. Example: --diag_perm 3 2 1 4 0")

    p.add_argument("--debug", type=int, default=0)

    args = p.parse_args()

    print(f"[Device] {DEVICE}")
    print(f"[CKPT] {args.ckpt}")

    model_type = infer_model_type_from_ckpt(args.ckpt, forced=args.model_type)
    print(f"[Model] {model_type}")

    model = load_diagnoser(args.ckpt, model_type=model_type, debug=args.debug)

    x = load_image_tensor(args.input)
    x_in = preprocess_for_diagnoser(x, img_size=int(args.diag_img_size), mode=args.diag_mode)

    print(f"[Patch] {args.diag_img_size} | crop= {args.diag_mode}")
    print(f"[Input] {args.input}")

    out = model(x_in)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        logits = out[0]
        m0 = out[1]
    elif torch.is_tensor(out):
        logits = out
        m0 = None
    elif isinstance(out, dict):
        logits = out.get("logits", out.get("s", out.get("score", None)))
        m0 = out.get("m0", out.get("map", None))
    else:
        raise RuntimeError(f"Unknown diagnoser output type: {type(out)}")

    if not torch.is_tensor(logits):
        raise RuntimeError("Diagnoser logits is not a tensor.")

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # raw logits -> sigmoid scores
    s0_sig_before = torch.sigmoid(logits[:, :5])

    # m0 stats
    if torch.is_tensor(m0) and m0.dim() == 4:
        m0 = m0[:, :5]
        m0_mean_logit = m0.mean(dim=(2, 3))
        m0_max_logit = m0.amax(dim=(2, 3))
        m0_mean_sig = torch.sigmoid(m0_mean_logit)
        m0_max_sig = torch.sigmoid(m0_max_logit)
    else:
        m0_mean_logit = None
        m0_max_logit = None
        m0_mean_sig = None
        m0_max_sig = None

    # apply permutation (scores + maps)
    perm = args.diag_perm
    if perm is not None:
        s0_sig_after = apply_perm_5(s0_sig_before, perm)
        if torch.is_tensor(m0) and m0.dim() == 4:
            m0 = apply_perm_5(m0, perm)
            m0_mean_logit = apply_perm_5(m0_mean_logit, perm)
            m0_max_logit = apply_perm_5(m0_max_logit, perm)
            m0_mean_sig = apply_perm_5(m0_mean_sig, perm)
            m0_max_sig = apply_perm_5(m0_max_sig, perm)
    else:
        s0_sig_after = s0_sig_before

    # print s0
    s_np = s0_sig_after.detach().cpu().numpy()[0]
    msg = "  ".join([f"{name}={s_np[i]:.3f}" for i, name in enumerate(ACTIONS)])
    print(f"[s0] {msg}")

    # pred action
    pred_id = int(np.argmax(s_np))
    print(f"[pred_action] {ACTIONS_AGENT[pred_id]}")

    # debug mapping print
    if int(args.debug) == 1:
        print("[DEBUG] raw logits:", logits.detach().cpu().numpy())
        print("[DEBUG] sigmoid(before perm) s0:", s0_sig_before.detach().cpu().numpy())
        print("[DEBUG] sigmoid(after  perm) s0:", s0_sig_after.detach().cpu().numpy())
        if perm is not None:
            print(f"[DEBUG] diag_perm={perm}")

    # map stats print
    if m0_mean_logit is not None:
        ml = m0_mean_logit.detach().cpu().numpy()[0]
        mx = m0_max_logit.detach().cpu().numpy()[0]
        ms = m0_mean_sig.detach().cpu().numpy()[0]
        xs = m0_max_sig.detach().cpu().numpy()[0]

        print("[m0_mean(logit)] " + "  ".join([f"{ACTIONS[i]}={ml[i]:+.3f}" for i in range(5)]))
        print("[m0_max (logit)] " + "  ".join([f"{ACTIONS[i]}={mx[i]:+.3f}" for i in range(5)]))
        print("[m0_mean(sig)]   " + "  ".join([f"{ACTIONS[i]}={ms[i]:.3f}" for i in range(5)]))
        print("[m0_max (sig)]   " + "  ".join([f"{ACTIONS[i]}={xs[i]:.3f}" for i in range(5)]))
    else:
        print("[m0] (no map output in this diagnoser)")

    print("[Done]")


if __name__ == "__main__":
    main()
