# scripts/train_toolbank.py
import os
import sys
import time
import random
from typing import List, Dict, DefaultDict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --------------------------------------------------
# Make project import-safe
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Imports
# --------------------------------------------------
from datasets.mixed_dataset import MixedDataset
from datasets.csd import CSDDataset
from datasets.rain100 import Rain100Dataset
from datasets.raindrop_day import DayRainDropDataset
from datasets.raindrop_night import NightRainDropDataset
from datasets.reside6k import RESIDE6KDataset

from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank, AdapterSpec

from models.planner.action_space import (
    A_DEDROP, A_DEBLUR, A_DESNOW, A_DERAIN, A_DEHAZE, A_HYBRID, A_STOP
)

# --------------------------------------------------
# Utils
# --------------------------------------------------
import yaml
import numpy as np
from PIL import Image


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Loss
# --------------------------------------------------
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


# --------------------------------------------------
# Paired Transform
# --------------------------------------------------
def _pil_to_tensor(img: Image.Image):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _random_crop_pair(inp, gt, patch):
    w, h = inp.size
    if w < patch or h < patch:
        scale = max(patch / w, patch / h)
        nw, nh = int(w * scale), int(h * scale)
        inp = inp.resize((nw, nh), Image.BILINEAR)
        gt = gt.resize((nw, nh), Image.BILINEAR)
        w, h = inp.size
    x = random.randint(0, w - patch)
    y = random.randint(0, h - patch)
    return inp.crop((x, y, x + patch, y + patch)), gt.crop((x, y, x + patch, y + patch))


def _augment_pair(inp, gt):
    if random.random() < 0.5:
        inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    k = random.randint(0, 3)
    if k > 0:
        inp = inp.rotate(90 * k, expand=True)
        gt = gt.rotate(90 * k, expand=True)
    return inp, gt


class PairedTransform:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

    def __call__(self, inp, gt):
        inp, gt = _random_crop_pair(inp, gt, self.patch_size)
        inp, gt = _augment_pair(inp, gt)
        return _pil_to_tensor(inp), _pil_to_tensor(gt)


# --------------------------------------------------
# Oracle Action
# --------------------------------------------------
def choose_oracle_action(
    degs: List[str],
    p_hybrid_when_multi: float = 0.7,
    p_force_desnow_in_multi: float = 0.3,
) -> str:
    d = [x.lower() for x in degs]
    s = set(d)

    cand = []
    if "drop" in s:
        cand.append(A_DEDROP)
    if "rain" in s:
        cand.append(A_DERAIN)
    if "haze" in s:
        cand.append(A_DEHAZE)
    if "snow" in s:
        cand.append(A_DESNOW)
    if "blur" in s:
        cand.append(A_DEBLUR)

    if len(cand) == 0:
        return A_HYBRID
    if len(cand) == 1:
        return cand[0]

    if random.random() < float(p_hybrid_when_multi):
        return A_HYBRID

    if (A_DESNOW in cand) and (random.random() < float(p_force_desnow_in_multi)):
        return A_DESNOW

    return random.choice(cand)


# --------------------------------------------------
# Custom Collate
# --------------------------------------------------
def react_ir_collate_fn(batch):
    return {
        "input": torch.stack([b["input"] for b in batch], dim=0),
        "gt": torch.stack([b["gt"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


# --------------------------------------------------
# LoRA Param Collection (DEDUP)
# --------------------------------------------------
def collect_lora_params(model: nn.Module):
    params = []
    seen = set()
    for name, p in model.named_parameters():
        # LoRAConv2d 내부의 base.weight는 제외하고, lora_A/lora_B만 학습
        if (".lora_A." in name) or (".lora_B." in name):
            if id(p) not in seen:
                p.requires_grad = True
                params.append(p)
                seen.add(id(p))
        else:
            # 혹시라도 base가 풀렸으면 다시 잠그기(보험)
            if ".base." in name:
                p.requires_grad = False
    return params


# --------------------------------------------------
# Action param cache (authoritative)
# --------------------------------------------------
def build_action_param_cache(toolbank: ToolBank) -> Dict[str, List[torch.nn.Parameter]]:
    cache: Dict[str, List[torch.nn.Parameter]] = {}
    for action, modules in toolbank.adapters.items():
        params = []
        seen = set()
        for m in modules:
            for p in m.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        cache[action] = params
    return cache


def grad_norm_for_action_params(action_param_cache: Dict[str, List[torch.nn.Parameter]], action: str) -> float:
    sq_sum = 0.0
    for p in action_param_cache.get(action, []):
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq_sum += float(g.float().pow(2).sum().item())
    return float(sq_sum ** 0.5)


def grad_norm_sum_except_params(
    action_param_cache: Dict[str, List[torch.nn.Parameter]],
    exclude_action: str,
    actions: List[str],
) -> float:
    sq_sum = 0.0
    for a in actions:
        if a == exclude_action:
            continue
        for p in action_param_cache.get(a, []):
            if p.grad is None:
                continue
            g = p.grad.detach()
            sq_sum += float(g.float().pow(2).sum().item())
    return float(sq_sum ** 0.5)


# --------------------------------------------------
# Δθ silence
# --------------------------------------------------
def build_action_theta0_snapshot_fp16(
    action_param_cache: Dict[str, List[torch.nn.Parameter]],
) -> Dict[str, List[torch.Tensor]]:
    snap: Dict[str, List[torch.Tensor]] = {}
    for a, params in action_param_cache.items():
        snap[a] = [p.detach().to("cpu", dtype=torch.float16).clone() for p in params]
    return snap


def silence_delta_theta_loss_mean(
    action_param_cache: Dict[str, List[torch.nn.Parameter]],
    theta0_snapshot_fp16_cpu: Dict[str, List[torch.Tensor]],
    active_action: str,
) -> torch.Tensor:
    device = None
    loss = None
    count = 0

    for a, params in action_param_cache.items():
        if a == active_action:
            continue
        t0_list = theta0_snapshot_fp16_cpu.get(a, None)
        if t0_list is None or len(t0_list) != len(params):
            continue

        for p, t0_cpu in zip(params, t0_list):
            if device is None:
                device = p.device
            t0 = t0_cpu.to(device=p.device, dtype=p.dtype, non_blocking=True)
            diff = (p - t0)
            term = torch.mean(diff * diff)
            loss = term if loss is None else (loss + term)
            count += 1

    if loss is None:
        return torch.tensor(
            0.0,
            device=(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")),
        )
    return loss / float(max(1, count))


# --------------------------------------------------
# DIAG A) optimizer param coverage / id match
# --------------------------------------------------
def _build_param_owner_name_map(model: nn.Module) -> Dict[int, str]:
    """
    id(param) -> "module_name.param_name" mapping for readable debugging.
    """
    m = {}
    for mod_name, mod in model.named_modules():
        for pn, p in mod.named_parameters(recurse=False):
            m[id(p)] = f"{mod_name}.{pn}" if mod_name else pn
    return m


def diag_optimizer_lora_coverage_once(
    toolbank: nn.Module,
    trainable_lora: List[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    action_param_cache: Dict[str, List[torch.nn.Parameter]],
    k_print: int = 5,
):
    # 1) counts
    opt_ids = set()
    opt_param_cnt = 0
    for gi, g in enumerate(optimizer.param_groups):
        ps = g.get("params", [])
        opt_param_cnt += len(ps)
        for p in ps:
            opt_ids.add(id(p))

    lora_ids = {id(p) for p in trainable_lora}

    cache_ids = set()
    for a, ps in action_param_cache.items():
        for p in ps:
            cache_ids.add(id(p))

    # 2) id set relations
    lora_not_in_opt = sorted(list(lora_ids - opt_ids))
    cache_not_in_opt = sorted(list(cache_ids - opt_ids))
    lora_not_in_cache = sorted(list(lora_ids - cache_ids))
    cache_not_in_lora = sorted(list(cache_ids - lora_ids))

    # 3) printable names
    name_map = _build_param_owner_name_map(toolbank)

    def _fmt(pid: int) -> str:
        return name_map.get(pid, "<unknown>")

    print("\n" + "-" * 78)
    print("[DIAG][OPT] optimizer param coverage check (ONCE)")
    print(f"[DIAG][OPT] len(trainable_lora) = {len(trainable_lora)}")
    print(f"[DIAG][OPT] optimizer.param_groups = {len(optimizer.param_groups)}")
    print(f"[DIAG][OPT] optimizer total param count (sum of groups) = {opt_param_cnt}")
    print(f"[DIAG][OPT] unique ids in optimizer = {len(opt_ids)}")
    print(f"[DIAG][OPT] unique ids in trainable_lora = {len(lora_ids)}")
    print(f"[DIAG][OPT] unique ids in action_param_cache(all) = {len(cache_ids)}")

    print(f"[DIAG][OPT] trainable_lora NOT in optimizer = {len(lora_not_in_opt)}")
    if len(lora_not_in_opt) > 0:
        for pid in lora_not_in_opt[: min(k_print, len(lora_not_in_opt))]:
            print(f"  - missing(lora->opt): id={pid} name={_fmt(pid)}")

    print(f"[DIAG][OPT] action_param_cache params NOT in optimizer = {len(cache_not_in_opt)}")
    if len(cache_not_in_opt) > 0:
        for pid in cache_not_in_opt[: min(k_print, len(cache_not_in_opt))]:
            print(f"  - missing(cache->opt): id={pid} name={_fmt(pid)}")

    print(f"[DIAG][OPT] trainable_lora NOT in action_param_cache = {len(lora_not_in_cache)}")
    if len(lora_not_in_cache) > 0:
        for pid in lora_not_in_cache[: min(k_print, len(lora_not_in_cache))]:
            print(f"  - mismatch(lora not in cache): id={pid} name={_fmt(pid)}")

    print(f"[DIAG][OPT] action_param_cache NOT in trainable_lora = {len(cache_not_in_lora)}")
    if len(cache_not_in_lora) > 0:
        for pid in cache_not_in_lora[: min(k_print, len(cache_not_in_lora))]:
            print(f"  - mismatch(cache not in lora): id={pid} name={_fmt(pid)}")

    # 4) sample 5 params from trainable_lora
    print(f"[DIAG][OPT] sample {min(k_print, len(trainable_lora))} params from trainable_lora:")
    for p in trainable_lora[: min(k_print, len(trainable_lora))]:
        pid = id(p)
        nm = name_map.get(pid, "<unknown>")
        print(
            f"  - name={nm} id={pid} requires_grad={bool(p.requires_grad)} "
            f"dtype={str(p.dtype)} device={str(p.device)}"
        )
    print("-" * 78 + "\n")


# --------------------------------------------------
# DIAG 1) scan backbone LoRA module states
# --------------------------------------------------
def diag_scan_lora_modules(backbone: nn.Module, limit: int = 8) -> str:
    """
    backbone 안의 LoRA 관련 모듈을 스캔해서,
    enable/scale/active_adapter 같은 상태가 실제로 바뀌는지 확인용 문자열 리턴.
    (프로젝트마다 속성명이 다를 수 있어 '있는 것만' 출력)
    """
    lines = []
    cnt = 0
    for name, m in backbone.named_modules():
        if hasattr(m, "is_lora") and bool(getattr(m, "is_lora")):
            fields = {}
            for k in [
                "enabled", "enable", "active", "active_adapter", "current_action",
                "scale", "scaling", "alpha", "rank", "merged"
            ]:
                if hasattr(m, k):
                    v = getattr(m, k)
                    try:
                        if torch.is_tensor(v):
                            v = float(v.detach().cpu().item())
                    except Exception:
                        pass
                    fields[k] = v
            lines.append(f"[LoRA] {name} :: {fields}")
            cnt += 1
            if cnt >= limit:
                break
    if cnt == 0:
        return "[LoRA] No modules found with attribute is_lora=True in backbone."
    return "\n".join(lines)


# --------------------------------------------------
# DIAG 2) track param updates (Δparam)
# --------------------------------------------------
def diag_pick_params(
    action_param_cache: Dict[str, List[torch.nn.Parameter]],
    actions: List[str],
    k_per_action: int = 1,
):
    picks = {}
    for a in actions:
        ps = action_param_cache.get(a, [])
        if len(ps) == 0:
            continue
        picks[a] = ps[:k_per_action]
    return picks


def diag_param_stats(p: torch.nn.Parameter):
    with torch.no_grad():
        w = p.detach()
        return float(w.float().norm().item()), float(w.float().abs().max().item())


# --------------------------------------------------
# ActionGate
# --------------------------------------------------
class ActionGate(nn.Module):
    def __init__(
        self,
        actions: List[str],
        init: float = 0.9,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.actions = list(actions)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.eps = float(eps)

        self.logits = nn.ParameterDict()

        init = float(init)
        init = float(np.clip(init, self.eps, 1.0 - self.eps))
        init_logit = float(np.log(init / (1.0 - init)))

        for a in self.actions:
            self.logits[a] = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def get_gate(self, action: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if action not in self.logits:
            base = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            base = torch.sigmoid(self.logits[action].to(device=device, dtype=dtype))
        g = self.clamp_min + (self.clamp_max - self.clamp_min) * base
        return g


class ToolBankWithActionGate(ToolBank):
    """
    apply(x, action) = stop_out + gate(action) * (act_out - stop_out)
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter_specs: Dict[str, AdapterSpec],
        device,
        debug: bool = False,
        gate_actions: List[str] = None,
        gate_init: float = 0.9,
        gate_clamp: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(backbone, adapter_specs, device=device, debug=debug)
        if gate_actions is None:
            gate_actions = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID]
        self.action_gate = ActionGate(
            actions=gate_actions,
            init=gate_init,
            clamp_min=float(gate_clamp[0]),
            clamp_max=float(gate_clamp[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _tb_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def apply(self, x: torch.Tensor, action: str):
        # base (STOP) - no grad needed
        self.activate_adapter(A_STOP)
        with torch.no_grad():
            out_base = self._tb_forward(x)

        if action == A_STOP:
            return out_base

        # action output (grad flows into LoRA + gate)
        self.activate_adapter(action)
        out_act = self._tb_forward(x)

        g = self.action_gate.get_gate(action, device=out_act.device, dtype=out_act.dtype)
        out = out_base + g * (out_act - out_base)
        return out

    def get_gate_value(self, action: str) -> float:
        with torch.no_grad():
            if action not in self.action_gate.logits:
                return 1.0
            v = float(torch.sigmoid(self.action_gate.logits[action]).detach().cpu().item())
            gv = self.action_gate.clamp_min + (self.action_gate.clamp_max - self.action_gate.clamp_min) * v
            return float(np.clip(gv, self.action_gate.clamp_min, self.action_gate.clamp_max))


# --------------------------------------------------
# DIAG delta stats: mean/max/rms (eps 제거: 0이면 0)
# --------------------------------------------------
@torch.no_grad()
def output_delta_stats_vs_stop(toolbank: ToolBankWithActionGate, x: torch.Tensor, action: str) -> Dict[str, float]:
    x1 = x[:1]
    out_stop = toolbank.apply(x1, A_STOP)
    out_act = toolbank.apply(x1, action)
    delta = (out_act - out_stop).float()
    mean_abs = float(delta.abs().mean().item())
    max_abs = float(delta.abs().max().item())
    rms = float(torch.sqrt(torch.mean(delta * delta)).item())  # NOTE: eps 더하지 않음
    return {"mean_abs": mean_abs, "max_abs": max_abs, "rms": rms}


# --------------------------------------------------
# Dataset Factory
# --------------------------------------------------
def build_single_dataset(root, cfg, tfm):
    t = cfg["type"]
    split = cfg.get("split", "train")
    if t == "CSDDataset":
        return CSDDataset(root=root, split=split, transform=tfm, debug=False)
    if t == "DayRainDropDataset":
        return DayRainDropDataset(root=root, split=split, transform=tfm, debug=False)
    if t == "NightRainDropDataset":
        return NightRainDropDataset(root=root, split=split, transform=tfm, debug=False)
    if t == "Rain100Dataset":
        return Rain100Dataset(root=root, split=split, transform=tfm, debug=False)
    if t == "RESIDE6KDataset":
        return RESIDE6KDataset(root=root, split=split, transform=tfm, debug=False)
    if t == "RESIDE6KDataset":
        return RESIDE6KDataset(root=root, split=split, transform=tfm, debug=False)
    raise ValueError(t)


def build_mixed_dataset(cfg, tfm):
    root = cfg["data_root"]
    mixed = cfg["mixed"]
    datasets = [build_single_dataset(root, dcfg, tfm) for dcfg in cfg["datasets"].values()]
    return MixedDataset(
        datasets=datasets,
        balance=mixed.get("balance", "sqrt"),
        epoch_length=int(mixed.get("epoch_length", 20000)),
        seed=int(mixed.get("seed", 123)),
        debug=True,
    )


# --------------------------------------------------
# Action helpers
# --------------------------------------------------
def build_cycle_actions(cfg_tools: dict) -> List[str]:
    cyc = cfg_tools.get("train", {}).get("balance_cycle", None)
    if cyc is None:
        return [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID]
    return list(cyc)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    cfg_ds = load_yaml(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    cfg_tools = load_yaml(os.path.join(PROJECT_ROOT, "configs", "tools.yaml"))

    set_seed(int(cfg_ds.get("mixed", {}).get("seed", 123)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    sanity_every = int(cfg_tools.get("train", {}).get("sanity_every", 200))
    diff_every = int(cfg_tools.get("train", {}).get("diff_every", 500))
    log_every = int(cfg_tools.get("train", {}).get("log_every", 50))

    # diag_steps: 강제 DIAG(=sanity+diff+추가 상태 출력) 찍는 step 리스트
    diag_steps = cfg_tools.get("train", {}).get("diag_steps", [20, 40])
    if not isinstance(diag_steps, list) or len(diag_steps) == 0:
        diag_steps = [20, 40]
    diag_steps = [int(s) for s in diag_steps]
    diag_step_set = set(diag_steps)
    print(f"[Diag] diag_steps={diag_steps}")

    p_hybrid_when_multi = float(cfg_tools.get("train", {}).get("p_hybrid_when_multi", 0.7))
    p_force_desnow_in_multi = float(cfg_tools.get("train", {}).get("p_force_desnow_in_multi", 0.3))

    balance_cycle = build_cycle_actions(cfg_tools)

    lambda_silence = float(cfg_tools.get("train", {}).get("lambda_silence", 0.0))
    silence_warmup_steps = int(cfg_tools.get("train", {}).get("silence_warmup_steps", 0))
    silence_every = int(cfg_tools.get("train", {}).get("silence_every", 1))

    gate_cfg = cfg_tools.get("train", {}).get("action_gate", {}) if isinstance(cfg_tools.get("train", {}), dict) else {}
    gate_init = float(gate_cfg.get("init", 0.9))
    gate_clamp_min = float(gate_cfg.get("clamp_min", 0.0))
    gate_clamp_max = float(gate_cfg.get("clamp_max", 1.0))
    gate_lr_scale = float(gate_cfg.get("lr_scale", 1.0))

    amp_cfg = cfg_tools.get("train", {}).get("amp_cfg", {}) if isinstance(cfg_tools.get("train", {}), dict) else {}
    scaler_init_scale = float(amp_cfg.get("init_scale", 2**12))
    scaler_growth_interval = int(amp_cfg.get("growth_interval", 200))

    lr = float(cfg_tools["train"]["lr"])
    weight_decay = float(cfg_tools["train"].get("weight_decay", 0.0))
    betas = cfg_tools["train"].get("betas", [0.9, 0.999])
    betas = (float(betas[0]), float(betas[1]))
    grad_clip = float(cfg_tools["train"].get("grad_clip", 0.0))
    use_amp = bool(cfg_tools["train"].get("amp", True))

    # Dataset
    tfm = PairedTransform(cfg_ds["train_patch"]["patch_size"])
    train_set = build_mixed_dataset(cfg_ds, tfm)

    loader = DataLoader(
        train_set,
        batch_size=cfg_ds["loader"]["batch_size"],
        shuffle=True,
        num_workers=cfg_ds["loader"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=react_ir_collate_fn,
    )

    # Model
    backbone = VETNet(dim=48, volterra_rank=4).to(device)
    adapter_specs = {a: AdapterSpec(**s) for a, s in cfg_tools["toolbank"]["adapters"].items()}

    # ToolBank + ActionGate
    toolbank = ToolBankWithActionGate(
        backbone,
        adapter_specs,
        device=device,
        debug=True,
        gate_actions=[A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID],
        gate_init=gate_init,
        gate_clamp=(gate_clamp_min, gate_clamp_max),
    ).to(device)

    # Freeze backbone (LoRA만 학습)
    if bool(cfg_tools.get("toolbank", {}).get("freeze_backbone", True)):
        for p in toolbank.backbone.parameters():
            p.requires_grad = False

    # Trainables: LoRA + ActionGate params
    trainable_lora = collect_lora_params(toolbank)
    trainable_gate = list(toolbank.action_gate.parameters())
    for p in trainable_gate:
        p.requires_grad = True

    lora_numel = sum(p.numel() for p in trainable_lora)
    gate_numel = sum(p.numel() for p in trainable_gate)
    print(f"[Trainable LoRA] {lora_numel/1e6:.2f} M params ({lora_numel} params)")
    print(f"[Trainable Gate] {gate_numel} params")

    # Optimizer (param groups so gate can have separate lr if needed)
    param_groups = [{"params": trainable_lora, "lr": lr}]
    if len(trainable_gate) > 0:
        param_groups.append({"params": trainable_gate, "lr": lr * gate_lr_scale})

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
    scaler = GradScaler("cuda", init_scale=scaler_init_scale, growth_interval=scaler_growth_interval) if use_amp else None

    # Action param cache + θ0 snapshot
    action_param_cache = build_action_param_cache(toolbank)
    theta0_snapshot = build_action_theta0_snapshot_fp16(action_param_cache)
    print("[SilenceΔθ] snapshot(theta0) captured (fp16, cpu).")

    # DIAG A) optimizer coverage check (ONCE at init)
    diag_optimizer_lora_coverage_once(
        toolbank=toolbank,
        trainable_lora=trainable_lora,
        optimizer=optimizer,
        action_param_cache=action_param_cache,
        k_print=5,
    )

    # DIAG 2) pick params + snapshot (per-action 1개씩)
    diag_actions = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID]
    diag_picks = diag_pick_params(action_param_cache, diag_actions, k_per_action=1)
    diag_snap: Dict[str, List[torch.Tensor]] = {}
    for a, ps in diag_picks.items():
        diag_snap[a] = [p.detach().clone().to("cpu", dtype=torch.float32) for p in ps]
    print("[DIAG] picked params per action:", {k: len(v) for k, v in diag_picks.items()})

    # Training cfg
    save_dir = cfg_tools["paths"]["ckpt_dir"]
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(cfg_tools["train"]["epochs"])

    print(f"[Train] batch_size={cfg_ds['loader']['batch_size']} epochs={epochs} lr={lr} seed={cfg_ds.get('mixed', {}).get('seed', 123)}")
    print(f"[OraclePolicy] p_hybrid_when_multi={p_hybrid_when_multi} p_force_desnow_in_multi={p_force_desnow_in_multi}")
    print(f"[ActionBalance] cycle={balance_cycle}")
    print(f"[SilenceΔθ] lambda_silence={lambda_silence} warmup={silence_warmup_steps} every={silence_every}")
    print(f"[ActionGate] init={gate_init} clamp=[{gate_clamp_min},{gate_clamp_max}] lr_scale={gate_lr_scale}")
    print(f"[AMP] use_amp={use_amp} scaler_init_scale={scaler_init_scale} growth_interval={scaler_growth_interval}")
    print(f"[Sanity] sanity_every={sanity_every} diff_every={diff_every} log_every={log_every}")

    global_step = 0
    toolbank.train()

    action_loss_sum: DefaultDict[str, float] = defaultdict(float)
    action_count: DefaultDict[str, int] = defaultdict(int)
    action_seen: DefaultDict[str, int] = defaultdict(int)

    prev_action: str = A_STOP
    t0 = time.time()

    steps_per_epoch = len(loader)
    print(f"[Train] steps_per_epoch={steps_per_epoch} (MixedDataset.__len__)")

    cycle_idx = 0
    sanity_actions = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID]

    for epoch in range(epochs):
        total = 0.0
        action_loss_sum.clear()
        action_count.clear()

        pbar = tqdm(loader, ncols=120)

        for it, batch in enumerate(pbar, start=1):
            global_step += 1

            x = batch["input"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            oracle_actions = [
                choose_oracle_action(
                    m["degradations"],
                    p_hybrid_when_multi=p_hybrid_when_multi,
                    p_force_desnow_in_multi=p_force_desnow_in_multi,
                )
                for m in batch["meta"]
            ]
            oracle_major = max(set(oracle_actions), key=oracle_actions.count)

            action = balance_cycle[cycle_idx % len(balance_cycle)]
            cycle_idx += 1

            mismatch = sum(1 for a in oracle_actions if a != action)
            mismatch_rate = float(mismatch) / float(max(1, len(oracle_actions)))
            oracle_hist = defaultdict(int)
            for a in oracle_actions:
                oracle_hist[a] += 1

            # Silence on/off
            use_silence = (lambda_silence > 0.0)
            if use_silence:
                if not (global_step >= max(0, silence_warmup_steps) and (silence_every <= 1 or (global_step % silence_every == 0))):
                    use_silence = False

            # Forward + loss
            if use_amp:
                with autocast(device_type="cuda"):
                    pred = toolbank.apply(x, action)
                loss_rec = charbonnier_loss(pred.float(), gt.float())
                loss = loss_rec
                if use_silence:
                    l_sil = silence_delta_theta_loss_mean(action_param_cache, theta0_snapshot, action)
                    loss = loss + (loss_rec.new_tensor(lambda_silence) * l_sil)
            else:
                pred = toolbank.apply(x, action)
                loss_rec = charbonnier_loss(pred, gt)
                loss = loss_rec
                if use_silence:
                    l_sil = silence_delta_theta_loss_mean(action_param_cache, theta0_snapshot, action)
                    loss = loss + (loss_rec.new_tensor(lambda_silence) * l_sil)

            # Backward / step
            optimizer.zero_grad(set_to_none=True)
            did_unscale = False

            def ensure_unscale():
                nonlocal did_unscale
                if (not did_unscale) and use_amp:
                    scaler.unscale_(optimizer)
                    did_unscale = True

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if grad_clip > 0:
                if use_amp:
                    ensure_unscale()
                torch.nn.utils.clip_grad_norm_(list(trainable_lora) + list(trainable_gate), grad_clip)

            # Decide diag/sanity/diff
            force_diag = (global_step in diag_step_set)
            do_sanity = (sanity_every > 0 and (global_step % sanity_every == 0))
            do_diff = (diff_every > 0 and (global_step % diff_every == 0))

            # DIAG 1) LoRA state scan BEFORE optimizer step
            if force_diag:
                print(f"[DIAG][lora_state][pre_step] step={global_step} action={action}\n{diag_scan_lora_modules(toolbank.backbone, limit=6)}")

            # SANITY grad stats (after backward; before step)
            if force_diag or do_sanity:
                tag = "DIAG" if force_diag else "SANITY"
                if use_amp:
                    try:
                        print(f"[{tag}][scaler] step={global_step} scale={scaler.get_scale():.1f}")
                    except Exception:
                        pass
                    ensure_unscale()

                max_g = 0.0
                cnt = 0
                for p in trainable_lora:
                    if p.grad is not None:
                        cnt += 1
                        max_g = max(max_g, float(p.grad.detach().abs().max().item()))

                print(f"[{tag}][grad_cnt] step={global_step} cnt={cnt}/{len(trainable_lora)}")
                print(f"[{tag}][grad_max] step={global_step} max_abs_grad={max_g:.3e}")

                g_sel = grad_norm_for_action_params(action_param_cache, action)
                g_non = grad_norm_sum_except_params(action_param_cache, action, sanity_actions)
                ratio = g_sel / (g_non + 1e-12)
                gate_val_now = toolbank.get_gate_value(action)
                print(
                    f"[{tag}][grad] step={global_step} sel={action} "
                    f"g_sel={g_sel:.3e} sum_g_non_sel={g_non:.3e} ratio={ratio:.3e} gate={gate_val_now:.3f}"
                )

            # Optimizer step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # DIAG 2) param Δ after optimizer step
            if force_diag or do_diff:
                tag = "DIAG" if force_diag else "SANITY"
                for a, ps in diag_picks.items():
                    for i, p in enumerate(ps):
                        w0 = diag_snap[a][i].to("cpu")
                        w1 = p.detach().to("cpu", dtype=torch.float32)
                        dw = (w1 - w0)
                        n0, m0 = float(w0.norm().item()), float(w0.abs().max().item())
                        nd, md = float(dw.norm().item()), float(dw.abs().max().item())
                        print(
                            f"[{tag}][paramΔ] step={global_step} act={a} idx={i} "
                            f"|w0|={n0:.3e} max|w0|={m0:.3e} |Δ|={nd:.3e} max|Δ|={md:.3e}"
                        )
                        diag_snap[a][i] = w1.clone()

            # DIAG/SANITY diff: Δout_vs_stop(mean/max/rms)
            if force_diag or do_diff:
                toolbank.eval()
                with torch.no_grad():
                    stats = output_delta_stats_vs_stop(toolbank, x, action)
                    tag = "DIAG" if force_diag else "SANITY"
                    print(
                        f"[{tag}][diff] step={global_step} act={action} "
                        f"Δout_vs_stop(mean_abs)={stats['mean_abs']:.3e} "
                        f"max_abs={stats['max_abs']:.3e} rms={stats['rms']:.3e}"
                    )
                toolbank.train()

            # stats
            loss_val = float(loss.detach().item())
            total += loss_val
            avg_loss = total / float(it)

            action_loss_sum[action] += loss_val
            action_count[action] += 1
            action_seen[action] += 1

            elapsed = time.time() - t0
            sw_flag = 1 if (prev_action != A_STOP and action != prev_action) else 0
            gate_val = toolbank.get_gate_value(action)

            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "act": action,
                    "m": f"{mismatch_rate:.2f}",
                    "sw": f"{sw_flag:d}",
                    "gate": f"{gate_val:.2f}",
                    "min": f"{elapsed/60:.1f}",
                },
                refresh=False,
            )

            if global_step == 1 or (global_step % log_every == 0):
                tqdm.write(
                    f"[BATCH][debug] epoch={epoch+1}/{epochs} it={it}/{steps_per_epoch} step={global_step} "
                    f"target={action} oracle_major={oracle_major} mismatch={mismatch}/{len(oracle_actions)} "
                    f"ok={(oracle_major==action)} gate={gate_val:.3f} oracle_hist={dict(oracle_hist)}"
                )

            prev_action = action

        # Epoch checkpoint
        epoch_loss = total / max(1, steps_per_epoch)
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}_loss{epoch_loss:.4f}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "step": global_step,
                "toolbank": toolbank.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"[CKPT] Saved: {ckpt_path}")

        # Epoch summary
        print(f"[Epoch {epoch+1}] Action-wise mean loss:")
        for a in sanity_actions:
            c = action_count.get(a, 0)
            gv = toolbank.get_gate_value(a)
            if c > 0:
                m = action_loss_sum[a] / c
                print(f"  - {a:<8} : count={c:5d} mean_loss={m:.4f} gate={gv:.3f}")
            else:
                print(f"  - {a:<8} : count=    0 mean_loss=NA gate={gv:.3f}")

        print("[Seen so far]", {a: action_seen.get(a, 0) for a in sanity_actions})

    print("[Train] Done ✅")


if __name__ == "__main__":
    main()
