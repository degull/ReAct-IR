# scripts/run_agent_llmA.py
import os
import sys
import json
import argparse
import importlib
import importlib.util
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# =========================
# Project import-safe
# =========================
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# IMPORTANT:
# eval_planner.py defines the trained action order as below.
ACTIONS = ["A_DEBLUR", "A_DERAIN", "A_DESNOW", "A_DEHAZE", "A_DEDROP"]
A2I = {a: i for i, a in enumerate(ACTIONS)}
I2A = {i: a for a, i in A2I.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Debug print helper
# =========================
def dprint(debug: int, *args, **kwargs):
    if int(debug) == 1:
        print(*args, **kwargs)


# ============================================================
# Robust dynamic importer (package import -> file scan fallback)
# ============================================================
def import_symbol_from_module(module_path: str, symbol: str):
    m = importlib.import_module(module_path)
    return getattr(m, symbol)


def load_module_from_file(py_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"spec_from_file_location failed: {py_path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore
    return m


def scan_python_files(root: str) -> List[str]:
    out = []
    for dp, _, fn in os.walk(root):
        if any(bad in dp for bad in ["__pycache__", ".git", ".venv", "venv", "site-packages"]):
            continue
        for f in fn:
            if f.endswith(".py"):
                out.append(os.path.join(dp, f))
    return out


def find_symbol_in_project(
    symbol: str,
    filename_hints: List[str],
    project_root: str = PROJECT_ROOT,
    max_scan_files: int = 5000,
) -> Tuple[str, str]:
    all_py = scan_python_files(project_root)
    if len(all_py) > max_scan_files:
        all_py = all_py[:max_scan_files]

    hints_lower = [h.lower() for h in filename_hints]

    # 1) filename hint pass
    for p in all_py:
        bn = os.path.basename(p).lower()
        if any(h in bn for h in hints_lower):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                if f"class {symbol}" in txt or f"def {symbol}" in txt:
                    return p, os.path.splitext(os.path.basename(p))[0]
            except Exception:
                continue

    # 2) full scan
    needle1 = f"class {symbol}"
    needle2 = f"def {symbol}"
    for p in all_py:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if needle1 in txt or needle2 in txt:
                return p, os.path.splitext(os.path.basename(p))[0]
        except Exception:
            continue

    raise ImportError(
        f"Could not find symbol '{symbol}' in project by scanning. "
        f"Tried hints={filename_hints}"
    )


def import_first_with_file_fallback(
    symbol: str,
    module_candidates: List[str],
    filename_hints: List[str],
) -> Any:
    last_err = None

    for mod in module_candidates:
        try:
            return import_symbol_from_module(mod, symbol)
        except Exception as e:
            last_err = e

    py_path, mod_hint = find_symbol_in_project(symbol, filename_hints, PROJECT_ROOT)
    try:
        m = load_module_from_file(py_path, f"_dyn_{mod_hint}")
        return getattr(m, symbol)
    except Exception as e:
        last_err = e

    raise ImportError(f"Failed to import symbol '{symbol}'. Last error: {last_err}")


def pick_class_fuzzy(module, must_contain: List[str]) -> Any:
    keys = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            lname = name.lower()
            ok = True
            for kw in must_contain:
                if kw.lower() not in lname:
                    ok = False
                    break
            if ok:
                keys.append(name)

    if not keys:
        classes = [n for n in dir(module) if isinstance(getattr(module, n), type)]
        raise ImportError(
            f"Could not find any class in module '{module.__name__}' "
            f"containing keywords={must_contain}. Available classes={classes}"
        )

    keys.sort(key=lambda x: len(x))
    return getattr(module, keys[0])


def import_diagnoser_map_fuzzy() -> Any:
    module_candidates = [
        "models.diagnoser_map",
        "models.diagnoser.diagnoser_map",
        "models.diagnoser",
    ]
    for mod in module_candidates:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "DiagnoserMap"):
                return getattr(m, "DiagnoserMap")
        except Exception:
            pass

    py_files = scan_python_files(PROJECT_ROOT)
    hint_files = []
    for p in py_files:
        bn = os.path.basename(p).lower()
        if any(k in bn for k in ["diagnoser_map", "train_diagnoser_map", "diagnoser"]):
            hint_files.append(p)

    search_list = hint_files + [p for p in py_files if p not in hint_files]

    for py_path in search_list:
        try:
            m = load_module_from_file(py_path, f"_dyn_{os.path.splitext(os.path.basename(py_path))[0]}")
            if hasattr(m, "DiagnoserMap"):
                return getattr(m, "DiagnoserMap")
            return pick_class_fuzzy(m, must_contain=["diagnoser", "map"])
        except Exception:
            continue

    raise ImportError("Failed to import DiagnoserMap by fuzzy scan.")


# =========================
# Load helpers
# =========================
def load_ckpt_state(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "net", "weights", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


# =========================
# Diagnoser (vit_map / cnn_map) - must match train_diagnoser.py
# =========================
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
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid = self.img_size // self.patch_size
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
        B, C, H, W = x.shape
        x = self.patch_embed(x)                  # (B,D,Hp,Wp)
        x = x.flatten(2).transpose(1, 2)         # (B,N,D)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)

        x = self.encoder(x)                      # (B,N,D)
        x = self.norm(x)

        patch_logits = self.map_head(x)          # (B,N,5)
        m0 = patch_logits.transpose(1, 2).reshape(B, 5, self.grid, self.grid)  # (B,5,Hp,Wp)

        logit_mean = m0.mean(dim=(2, 3))
        logit_max  = m0.amax(dim=(2, 3))
        logits = 0.5 * (logit_mean + logit_max)  # (B,5)
        return logits, m0


class CNNMapDiagnoser(nn.Module):
    def __init__(self, num_labels: int = 5, width: int = 64):
        super().__init__()
        w = int(width)
        self.img_size = 256  # just for compatibility; actual input can be resized externally
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


def load_diagnoser_from_ckpt(ckpt_path: str, device: str = DEVICE, strict: bool = True) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Diagnoser ckpt must be dict, got {type(ckpt)}")

    # model type
    model_type = ckpt.get("model", None)
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg", {}), dict) else {}

    if model_type is None:
        model_type = cfg.get("model", None)
    if model_type is None:
        raise RuntimeError("Cannot find diagnoser model type in ckpt. Expected ckpt['model'] or ckpt['cfg']['model'].")

    model_type = str(model_type)

    # build model with exact hyperparams used in train_diagnoser.py
    if model_type == "vit_map":
        img_size   = int(cfg.get("patch", 256))
        patch_size = int(cfg.get("vit_patch", 16))
        dim        = int(cfg.get("vit_dim", 384))
        depth      = int(cfg.get("vit_depth", 6))
        heads      = int(cfg.get("vit_heads", 6))
        model = ViTMapDiagnoser(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=dim,
            depth=depth,
            num_heads=heads,
            num_labels=5,
        )
    elif model_type == "cnn_map":
        width = int(cfg.get("cnn_width", 64))
        model = CNNMapDiagnoser(num_labels=5, width=width)
    else:
        raise RuntimeError(f"Unknown diagnoser model_type='{model_type}'. Expected 'vit_map' or 'cnn_map'.")

    # state dict
    sd = ckpt.get("state_dict", None)
    if sd is None:
        sd = ckpt.get("model_state_dict", None)
    if sd is None:
        sd = load_ckpt_state(ckpt_path)

    model.load_state_dict(sd, strict=bool(strict))
    model.to(device).eval()
    return model


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t.to(DEVICE)


def save_image(t: torch.Tensor, path: str):
    t = t.clamp(0, 1)
    img = transforms.ToPILImage()(t.squeeze(0).detach().cpu())
    img.save(path)


# =========================
# Diagnoser-only preprocessing (fix img_size assert)
# =========================
def preprocess_for_diagnoser(x: torch.Tensor, img_size: int, mode: str = "resize") -> torch.Tensor:
    if not torch.is_tensor(x) or x.dim() != 4:
        raise ValueError(f"x must be 4D [B,C,H,W], got type={type(x)}, shape={getattr(x,'shape',None)}")

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
            return x[:, :, top:top + img_size, left:left + img_size]
        return F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    raise ValueError(f"Unknown diag_mode='{mode}'. Use 'resize' or 'center_crop'.")


def get_diagnoser_img_size(diagnoser: nn.Module, fallback: int = 256) -> int:
    for k in ["img_size", "image_size", "input_size", "im_size"]:
        if hasattr(diagnoser, k):
            v = getattr(diagnoser, k)
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except Exception:
                pass
    return int(fallback)


# =========================
# Feature builder EXACTLY like eval_planner.build_feat
# state_dim = 17 = s0(5) + m0_mean(5) + m0_max(5) + bpsnr + bssim
# =========================
def _nan_to_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def extract_s0_m0_from_diagnoser_output(
    out: Any,
    s_t_fallback: torch.Tensor,
    debug: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      s0      : [B,5]
      m0_mean : [B,5]
      m0_max  : [B,5]

    Supports diagnoser output formats:
      1) tensor logits: out = [B,5]
      2) tuple/list: out = (logits[B,5], m0[B,5,h,w])
      3) dict: out can contain {s/score/s0} and/or {m0/map/...} and/or {m0_mean,m0_max}
    """
    # ---- s0 fallback ----
    s0 = s_t_fallback
    if s0.dim() == 1:
        s0 = s0.unsqueeze(0)
    s0 = s0[:, :5].contiguous()

    B = s0.shape[0]
    zeros = torch.zeros((B, 5), device=s0.device, dtype=s0.dtype)
    m0_mean = zeros
    m0_max = zeros

    # (A) tuple/list output: (logits, m0)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        logits = out[0]
        m0 = out[1]

        if torch.is_tensor(logits):
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            s0 = logits[:, :5].to(device=s0.device, dtype=s0.dtype).contiguous()

        if torch.is_tensor(m0) and m0.dim() == 4:
            mm = m0[:, :5, :, :].to(device=s0.device, dtype=s0.dtype)
            m0_mean = mm.mean(dim=(2, 3))
            m0_max = mm.amax(dim=(2, 3))

        s0 = _nan_to_zero(s0)
        m0_mean = _nan_to_zero(m0_mean)
        m0_max = _nan_to_zero(m0_max)

        if int(debug) == 1:
            if torch.allclose(m0_mean, zeros) and torch.allclose(m0_max, zeros):
                dprint(1, "[DEBUG][Feat17] tuple out but m0 stats are zeros. Check m0 shape/channels.")
        return s0, m0_mean, m0_max

    # (B) dict output
    if isinstance(out, dict):
        # direct stats
        if "m0_mean" in out and torch.is_tensor(out["m0_mean"]):
            t = out["m0_mean"]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            m0_mean = t[:, :5].to(device=s0.device, dtype=s0.dtype).contiguous()

        if "m0_max" in out and torch.is_tensor(out["m0_max"]):
            t = out["m0_max"]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            m0_max = t[:, :5].to(device=s0.device, dtype=s0.dtype).contiguous()

        # map candidates
        map_keys = ["m0", "map", "maps", "m", "heatmap", "score_map", "degradation_map"]
        m0 = None
        for k in map_keys:
            if k in out and torch.is_tensor(out[k]):
                m0 = out[k]
                break

        # sometimes nested dicts
        if m0 is None:
            for kk, vv in out.items():
                if isinstance(vv, dict):
                    for k in map_keys:
                        if k in vv and torch.is_tensor(vv[k]):
                            m0 = vv[k]
                            break
                if m0 is not None:
                    break

        # compute spatial stats
        if torch.is_tensor(m0) and m0.dim() == 4:
            mm = m0[:, :5, :, :]
            m0_mean = mm.mean(dim=(2, 3))
            m0_max = mm.amax(dim=(2, 3))

    # (C) tensor-only output: no map/stats -> zeros for m0
    s0 = _nan_to_zero(s0)
    m0_mean = _nan_to_zero(m0_mean)
    m0_max = _nan_to_zero(m0_max)

    if int(debug) == 1:
        if torch.allclose(m0_mean, zeros) and torch.allclose(m0_max, zeros):
            dprint(
                1,
                "[DEBUG][Feat17] m0_mean/m0_max are ALL ZEROS (Diagnoser did not provide maps/stats). "
                "Planner may be less reliable unless diagnoser outputs m0."
            )

    return s0, m0_mean, m0_max


def build_planner_feat17(
    s0: torch.Tensor,
    m0_mean: torch.Tensor,
    m0_max: torch.Tensor,
    baseline_psnr: float,
    baseline_ssim: float,
) -> torch.Tensor:
    if s0.dim() == 1:
        s0 = s0.unsqueeze(0)
    if m0_mean.dim() == 1:
        m0_mean = m0_mean.unsqueeze(0)
    if m0_max.dim() == 1:
        m0_max = m0_max.unsqueeze(0)

    B = s0.shape[0]
    bpsnr = torch.full((B, 1), float(baseline_psnr), device=s0.device, dtype=s0.dtype)
    bssim = torch.full((B, 1), float(baseline_ssim), device=s0.device, dtype=s0.dtype)

    feat = torch.cat([s0[:, :5], m0_mean[:, :5], m0_max[:, :5], bpsnr, bssim], dim=1)
    assert feat.shape[1] == 17, f"feat17 must be 17-D, got {feat.shape}"
    feat = _nan_to_zero(feat)
    return feat


# =========================
# ValueHead(21) input builder (train_valuehead.py compatible)
# =========================
def build_value_x21_trainstyle(
    s0: torch.Tensor,
    m0_mean: torch.Tensor,
    m0_max: torch.Tensor,
    action: str,
    scale: float,
) -> torch.Tensor:
    """
    EXACTLY matches scripts/train_valuehead.py

    x = state_feat(15) + action_onehot(5) + scale(1) => 21
      state_feat(15) = [s0(5), m0_mean(5), m0_max(5)]
      action_onehot(5) uses ACTIONS order
      scale is last dim
    """
    if s0.dim() == 1:
        s0 = s0.unsqueeze(0)
    if m0_mean.dim() == 1:
        m0_mean = m0_mean.unsqueeze(0)
    if m0_max.dim() == 1:
        m0_max = m0_max.unsqueeze(0)

    B = s0.shape[0]
    device = s0.device
    dtype = s0.dtype

    # state_feat(15)
    state15 = torch.cat([s0[:, :5], m0_mean[:, :5], m0_max[:, :5]], dim=1)  # [B,15]

    # action one-hot(5)
    aoh = torch.zeros((B, len(ACTIONS)), device=device, dtype=dtype)
    aid = ACTIONS.index(action)
    aoh[:, aid] = 1.0

    # scale(1)
    sc = torch.full((B, 1), float(scale), device=device, dtype=dtype)

    x21 = torch.cat([state15, aoh, sc], dim=1)  # [B,21]
    assert x21.shape[1] == 21, f"x21 must be 21-D, got {x21.shape}"
    return _nan_to_zero(x21)


# =========================
# ValueHeadMLP (FIXED: same as train_valuehead.py)
# =========================
class ValueHeadMLP(nn.Module):
    def __init__(self, in_dim: int = 21, hidden: int = 256, depth: int = 3, dropout: float = 0.1, out_dim: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        h = int(hidden)
        dep = int(depth)
        do = float(dropout)

        for _ in range(dep):
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(do)]
            d = h
        layers += [nn.Linear(d, int(out_dim))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_valuehead_from_ckpt(ckpt_path: str, device: str = DEVICE) -> nn.Module:
    """
    Load valuehead ckpt saved by scripts/train_valuehead.py
    Expected ckpt keys: { 'model': state_dict, 'args': {...} } or similar.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"ValueHead ckpt must be dict, got {type(ckpt)}")

    # --- get args (train_valuehead.py typically stores args) ---
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = {}

    hidden = int(args.get("hidden", 256))
    depth = int(args.get("depth", 3))
    dropout = float(args.get("dropout", 0.1))

    model = ValueHeadMLP(in_dim=21, hidden=hidden, depth=depth, dropout=dropout, out_dim=2)

    # --- get state dict ---
    sd = None
    # prefer exact "model" key (train_valuehead.py)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    else:
        sd = load_ckpt_state(ckpt_path)

    # strict=True is key
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


# =========================
# Stop rules
# =========================
class LLMPlannerA:
    """
    Base chooser uses best candidate (max value).
    Spread stop is kept as auxiliary.
    """
    def __init__(self, stop_delta: float = 0.02, min_steps: int = 1):
        self.stop_delta = float(stop_delta)
        self.min_steps = int(min_steps)

    def choose(self, candidates: List[Dict[str, Any]], history_len: int) -> Dict[str, Any]:
        best = candidates[0]
        worst = candidates[-1]
        spread = float(best["value"] - worst["value"])

        stop = False
        if history_len >= self.min_steps:
            stop = (spread < self.stop_delta)

        return {
            "action": best["action"],
            "scale": float(best["scale"]),
            "stop": bool(stop),
            "reason": f"picked best (V_hat={best['value']:.4f}), spread={spread:.4f}, stop_delta={self.stop_delta:.4f}"
        }


class SafetyOverride:
    def __init__(self, scale_min=0.0, scale_max=1.0):
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def apply(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        decision["scale"] = float(np.clip(decision["scale"], self.scale_min, self.scale_max))
        if decision["action"] not in ACTIONS:
            decision["action"] = "A_DEDROP"
            decision["reason"] += " | fallback: unknown action"
        return decision


# =========================
# ToolBank robust caller
# =========================
def _public_methods(obj) -> List[str]:
    out = []
    for n in dir(obj):
        if n.startswith("_"):
            continue
        try:
            attr = getattr(obj, n)
        except Exception:
            continue
        if callable(attr):
            out.append(n)
    out.sort()
    return out


def call_toolbank(toolbank: nn.Module, x: torch.Tensor, action: str, scale: float) -> torch.Tensor:
    action_id = ACTIONS.index(action)

    direct_methods = [
        "run", "apply_tool", "apply_action", "infer", "forward_tool", "forward_with_tool",
        "restore", "execute", "step"
    ]
    for name in direct_methods:
        if hasattr(toolbank, name):
            fn = getattr(toolbank, name)
            try:
                y = fn(x, action, float(scale))
                if torch.is_tensor(y):
                    return y
            except Exception:
                pass
            try:
                y = fn(x, action=action, scale=float(scale))
                if torch.is_tensor(y):
                    return y
            except Exception:
                pass
            try:
                y = fn(x, action_id, float(scale))
                if torch.is_tensor(y):
                    return y
            except Exception:
                pass
            try:
                y = fn(x, action_id=action_id, scale=float(scale))
                if torch.is_tensor(y):
                    return y
            except Exception:
                pass

    setter_patterns = [
        ("set_tool", [(action, float(scale)), (action_id, float(scale))]),
        ("activate", [(action, float(scale)), (action_id, float(scale))]),
        ("use", [(action, float(scale)), (action_id, float(scale))]),
        ("select", [(action,), (action_id,)]),
        ("set_action", [(action,), (action_id,)]),
        ("set_adapter", [(action,), (action_id,)]),
        ("set_lora", [(action,), (action_id,)]),
        ("load_lora", [(action,), (action_id,)]),
    ]
    scale_setters = ["set_scale", "set_alpha", "set_strength", "set_weight"]

    for setter_name, arg_list in setter_patterns:
        if not hasattr(toolbank, setter_name):
            continue
        setter = getattr(toolbank, setter_name)
        for args in arg_list:
            try:
                setter(*args)
                for sname in scale_setters:
                    if hasattr(toolbank, sname):
                        try:
                            getattr(toolbank, sname)(float(scale))
                        except Exception:
                            pass
                y = toolbank(x)
                if torch.is_tensor(y):
                    return y
            except Exception:
                continue

    try:
        y = toolbank(x)
        if torch.is_tensor(y):
            return y
    except Exception:
        pass

    methods = _public_methods(toolbank)
    raise RuntimeError(
        "Failed to apply ToolBank with any known calling convention.\n"
        f"ToolBank public callables: {methods}\n"
        "Hint: check your ToolBank implementation for the correct API, e.g.\n"
        "- toolbank.set_tool(action, scale); y = toolbank(x)\n"
        "- y = toolbank.run(x, action, scale)\n"
        "- y = toolbank.apply_tool(x, action, scale)\n"
    )


# =========================
# Candidate builder
# =========================
def planner_forward_logits(planner_net: nn.Module, feat17: torch.Tensor) -> torch.Tensor:
    return planner_net(feat17)


def valuehead_forward(
    value_head: nn.Module,
    s0: torch.Tensor,
    m0_mean: torch.Tensor,
    m0_max: torch.Tensor,
    action: str,
    scale: float,
) -> float:
    """
    Train-time identical input:
      x21 = [state15, action_onehot(5), scale]
    Output:
      y = [d_psnr, d_ssim]
    We convert it to scalar value for ranking.
    """
    x21 = build_value_x21_trainstyle(
        s0=s0, m0_mean=m0_mean, m0_max=m0_max,
        action=action, scale=float(scale)
    )

    with torch.no_grad():
        y = value_head(x21)  # [B,2]
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.detach()

        dpsnr = float(y[0, 0].item())
        dssim = float(y[0, 1].item())

        # scalarize
        value = dpsnr + 100.0 * dssim
        return float(value)

def build_candidates(
    feat17: torch.Tensor,
    s0: torch.Tensor,
    m0_mean: torch.Tensor,
    m0_max: torch.Tensor,
    planner_net: nn.Module,
    value_head: nn.Module,
    scales: List[float],
    top_k: int,
    step: int,
    max_steps: int,
    debug: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:

    with torch.no_grad():
        logits = planner_forward_logits(planner_net, feat17)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        logits_np = logits.detach().cpu().numpy()[0]

    action_scores = list(zip(ACTIONS, probs, logits_np))
    action_scores.sort(key=lambda x: -x[1])
    keep = action_scores[:max(1, min(top_k, len(ACTIONS)))]

    if int(debug) == 1:
        dprint(1, "[DEBUG][Planner] top probs (idx, action, prob, logit):")
        for a, p, lg in keep:
            dprint(1, f"  {A2I[a]:02d}  {a:8s} prob={p:.4f}  logit={lg:.4f}")

    # ------------------------------------------------------------
    # Diagnoser prior (action guidance)
    # s0 is assumed to be sigmoid-ed scores in [0,1] already.
    # ------------------------------------------------------------
    LABELS = ["blur", "rain", "snow", "haze", "drop"]
    LABEL2ACTION = {
        "blur": "A_DEBLUR",
        "rain": "A_DERAIN",
        "snow": "A_DESNOW",
        "haze": "A_DEHAZE",
        "drop": "A_DEDROP",
    }

    # dominant label by diagnoser score
    dom_idx = int(torch.argmax(s0[0]).item())
    dom_label = LABELS[dom_idx]
    prior_action = LABEL2ACTION[dom_label]

    prior_bonus = 0.8  # 튜닝 파라미터 (0.3~2.0 사이에서 시작 추천)
    prior_other_penalty = 0.2  # optional small penalty

    candidates = []
    for action, _p, _lg in keep:
        for sc in scales:
            v = valuehead_forward(
                value_head=value_head,
                s0=s0,
                m0_mean=m0_mean,
                m0_max=m0_max,
                action=action,
                scale=float(sc),
            )

            # apply diagnoser-guided bonus/penalty
            if action == prior_action:
                v = v + prior_bonus
            else:
                v = v - prior_other_penalty

            candidates.append({"action": action, "scale": float(sc), "value": float(v)})

    candidates.sort(key=lambda x: -x["value"])
    return candidates[:top_k], probs, logits_np


# =========================
# Instantiate with flexible constructors
# =========================
def instantiate_backbone(VETNetClass):
    try:
        return VETNetClass()
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate VETNet(): {e}")


def instantiate_planner(PlannerActionNetClass):
    try:
        return PlannerActionNetClass()
    except Exception:
        try:
            return PlannerActionNetClass(num_actions=len(ACTIONS))
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate PlannerActionNet(): {e}")


# =========================
# Main
# =========================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------
    # Import symbols robustly
    # -------------------------
    VETNet = import_first_with_file_fallback(
        symbol="VETNet",
        module_candidates=[
            "models.backbone.vetnet",
            "models.backbone.vetnet_backbone",
            "models.vetnet",
        ],
        filename_hints=["vetnet.py", "vetnet_backbone.py", "backbone"],
    )

    ToolBank = import_first_with_file_fallback(
        symbol="ToolBank",
        module_candidates=[
            "models.toolbank.toolbank",
            "models.toolbank",
        ],
        filename_hints=["toolbank.py", "toolbank"],
    )

    PlannerActionNet = import_first_with_file_fallback(
        symbol="PlannerActionNet",
        module_candidates=[
            "models.planner_action.planner_action",
            "models.planner_action",
            "models.planner.planner_action",
            "scripts.eval_planner",
        ],
        filename_hints=["planner_action.py", "planner_action_only.py", "planner", "eval_planner.py"],
    )

    # -------------------------
    # Load models
    # -------------------------
    backbone = instantiate_backbone(VETNet).to(DEVICE).eval()
    backbone.load_state_dict(load_ckpt_state(args.backbone_ckpt), strict=False)

    # Diagnoser: load once (strict=True)
    diagnoser = load_diagnoser_from_ckpt(args.diagnoser_ckpt, device=DEVICE, strict=True)

    try:
        toolbank = ToolBank(backbone=backbone, lora_root=args.toolbank_lora_root).to(DEVICE).eval()
    except Exception:
        try:
            toolbank = ToolBank(backbone, args.toolbank_lora_root).to(DEVICE).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate ToolBank with both signatures: {e}")

    # ValueHead: load EXACT train_valuehead model (strict=True)
    valuehead = load_valuehead_from_ckpt(args.valuehead_ckpt, device=DEVICE)

    # Planner: keep your existing load (strict=False for now)
    planner = instantiate_planner(PlannerActionNet).to(DEVICE).eval()
    planner.load_state_dict(load_ckpt_state(args.planner_ckpt), strict=False)

    llm = LLMPlannerA(stop_delta=args.stop_delta, min_steps=args.min_steps)
    safety = SafetyOverride(scale_min=args.scale_min, scale_max=args.scale_max)

    # -------------------------
    # Run loop
    # -------------------------
    x = load_image(args.input)
    history: List[Dict[str, Any]] = []

    diag_size = get_diagnoser_img_size(diagnoser, fallback=args.diag_img_size)

    print(f"[ToolBank] injected LoRA into (see ToolBank print above if any)")
    print(f"[Input] {args.input}")
    print(f"[Input] x shape = {tuple(x.shape)}")
    print(f"[Diagnoser] expected img_size = {diag_size} (mode={args.diag_mode})")
    print(f"[Planner]  expected state_dim = 17 (fixed by eval_planner.build_feat)")
    print(f"[ValueHead] expected state_dim = 21 (train_valuehead.py compatible; strict=True)")
    print(f"[StopRule] spread_stop_delta={args.stop_delta}, min_steps={args.min_steps}, "
          f"score_thresh={args.stop_score_thresh}, patience={args.stop_patience}, eps={args.stop_improve_eps}")

    # stop bookkeeping
    best_value_prev: Optional[float] = None
    no_improve = 0

    for step in range(args.max_steps):
        print(f"\n=== Step {step} ===")

        # 1) Diagnose (on resized/cropped image)
        x_diag = preprocess_for_diagnoser(x, img_size=int(diag_size), mode=args.diag_mode)

        with torch.no_grad():
            out = diagnoser(x_diag)

            # s_t extraction for score_max (works for tensor / tuple / dict)
            if isinstance(out, (tuple, list)):
                s_t = out[0]
            elif isinstance(out, dict):
                s_t = out.get("s", out.get("score", None))
                if s_t is None:
                    s_t = out.get("s0", None)
                if s_t is None:
                    raise RuntimeError("Diagnoser dict output has no key in {s, score, s0}.")
            else:
                s_t = out

        if not torch.is_tensor(s_t):
            raise RuntimeError(f"Diagnoser output is not tensor. type={type(s_t)}")
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)

        # Debug: raw/sigmoid/softmax
        if args.debug == 1:
            s_raw = s_t.detach()
            dprint(1, "[DEBUG][Diagnoser] raw:", s_raw.cpu().numpy())
            dprint(1, "[DEBUG][Diagnoser] sigmoid:", torch.sigmoid(s_raw).cpu().numpy())
            dprint(1, "[DEBUG][Diagnoser] softmax:", torch.softmax(s_raw, dim=-1).cpu().numpy())

        # 2) Build feat17
        s0, m0_mean, m0_max = extract_s0_m0_from_diagnoser_output(out, s_t, debug=args.debug)

        # normalize to probability space
        s0 = torch.sigmoid(s0)
        m0_mean = torch.sigmoid(m0_mean)
        m0_max = torch.sigmoid(m0_max)

        feat17 = build_planner_feat17(
            s0=s0,
            m0_mean=m0_mean,
            m0_max=m0_max,
            baseline_psnr=float(args.baseline_psnr),
            baseline_ssim=float(args.baseline_ssim),
        )

        if args.debug == 1:
            dprint(1, f"[DEBUG][Feat17] feat17 shape={tuple(feat17.shape)}")
            dprint(1, f"[DEBUG][Feat17] s0      = {s0.detach().cpu().numpy()}")
            dprint(1, f"[DEBUG][Feat17] m0_mean = {m0_mean.detach().cpu().numpy()}")
            dprint(1, f"[DEBUG][Feat17] m0_max  = {m0_max.detach().cpu().numpy()}")
            dprint(1, f"[DEBUG][Feat17] baseline_psnr={args.baseline_psnr:.4f}, baseline_ssim={args.baseline_ssim:.6f}")

        # score_max for stop rule A
        score_max = float(torch.sigmoid(s_t.detach()).max().item())

        # 3) Candidates
        candidates, probs, logits_np = build_candidates(
            feat17=feat17,
            s0=s0,
            m0_mean=m0_mean,
            m0_max=m0_max,
            planner_net=planner,
            value_head=valuehead,
            scales=args.scales,
            top_k=args.top_k,
            step=step,
            max_steps=args.max_steps,
            debug=args.debug,
        )

        print("Top candidates:", candidates[:min(5, len(candidates))])

        # Debug: full action×scale table
        if args.debug == 1:
            dprint(1, "[DEBUG][ValueHead] action×scale value table:")
            for a in ACTIONS:
                row = []
                for sc in args.scales:
                    v = valuehead_forward(
                        value_head=valuehead,
                        s0=s0,
                        m0_mean=m0_mean,
                        m0_max=m0_max,
                        action=a,
                        scale=float(sc),
                    )
                    row.append((float(sc), float(v)))
                row_str = ", ".join([f"{s:.2f}:{v:+.4f}" for s, v in row])
                dprint(1, f"  {a}: {row_str}")

        # 4) Choose + stop rules
        decision = llm.choose(candidates, history_len=len(history))

        # A) diagnoser 기반 stop
        if score_max < args.stop_score_thresh and step >= args.min_steps:
            decision["stop"] = True
            decision["reason"] += f" | stop: score_max={score_max:.4f} < {args.stop_score_thresh:.4f}"

        # B) valuehead best value no improve
        best_v = float(candidates[0]["value"]) if len(candidates) > 0 else -1e18
        if best_value_prev is None:
            best_value_prev = best_v
            no_improve = 0
        else:
            if best_v > best_value_prev + args.stop_improve_eps:
                best_value_prev = best_v
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.stop_patience and step >= args.min_steps:
                    decision["stop"] = True
                    decision["reason"] += f" | stop: no_improve={no_improve}/{args.stop_patience} (eps={args.stop_improve_eps})"

        # 5) Safety
        decision = safety.apply(decision)
        print("Decision:", decision)

        if decision["stop"]:
            print("Stop triggered.")
            break

        # 6) Apply tool
        with torch.no_grad():
            x = call_toolbank(toolbank, x, decision["action"], decision["scale"])

        # 7) Save
        save_path = os.path.join(
            args.out_dir,
            f"step_{step:02d}_{decision['action']}_{decision['scale']:.2f}.png"
        )
        save_image(x, save_path)
        history.append(decision)

    # save trace
    trace_path = os.path.join(args.out_dir, "trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Trace saved to {trace_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--backbone_ckpt", default="E:/ReAct-IR/checkpoints/backbone/best_backbone.pth")
    p.add_argument("--diagnoser_ckpt", default="E:/ReAct-IR/checkpoints/diagnoser_1/diagnoser_map_best.pth")
    p.add_argument("--toolbank_lora_root", default="E:/ReAct-IR/checkpoints/toolbank_lora")
    p.add_argument("--valuehead_ckpt", default="E:/ReAct-IR/checkpoints/valuehead/valuehead_best.pth")
    p.add_argument("--planner_ckpt", default="E:/ReAct-IR/checkpoints/planner_action/planner_action_best.pth")

    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])

    p.add_argument("--scale_min", type=float, default=0.0)
    p.add_argument("--scale_max", type=float, default=1.0)

    # Diagnoser options
    p.add_argument("--diag_mode", type=str, default="resize", choices=["resize", "center_crop"])
    p.add_argument("--diag_img_size", type=int, default=256)

    # Baseline metrics for feat17 (eval_planner uses psnr_in/ssim_in, but test-time has no GT)
    p.add_argument("--baseline_psnr", type=float, default=0.0, help="baseline psnr_in used in feat17 (default 0)")
    p.add_argument("--baseline_ssim", type=float, default=0.0, help="baseline ssim_in used in feat17 (default 0)")

    # Spread stop (aux)
    p.add_argument("--stop_delta", type=float, default=0.02)
    p.add_argument("--min_steps", type=int, default=1)

    # (A)(B) stop rules
    p.add_argument("--stop_score_thresh", type=float, default=0.10,
                   help="stop if max(diagnoser score) < this threshold")
    p.add_argument("--stop_patience", type=int, default=2,
                   help="stop if best value doesn't improve for N consecutive steps")
    p.add_argument("--stop_improve_eps", type=float, default=1e-3,
                   help="minimum improvement in best value to reset patience")

    # Debug
    p.add_argument("--debug", type=int, default=0)

    args = p.parse_args()
    main(args)
