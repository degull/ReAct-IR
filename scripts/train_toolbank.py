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
    A_DEDROP, A_DEBLUR, A_DESNOW, A_DERAIN, A_DEHAZE, A_STOP
)

# MultiLoRA wrappers
from models.toolbank.lora import MultiLoRAConv2d, MultiLoRALinear

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
# Oracle Action (NO HYBRID)
# --------------------------------------------------
def choose_oracle_action_single(degs: List[str]) -> str:
    """
    NO HYBRID policy:
      - if multiple degs exist, pick one (priority or random).
    """
    d = [x.lower() for x in degs]
    s = set(d)

    # priority example (you can change):
    if "drop" in s:
        return A_DEDROP
    if "rain" in s:
        return A_DERAIN
    if "haze" in s:
        return A_DEHAZE
    if "snow" in s:
        return A_DESNOW
    if "blur" in s:
        return A_DEBLUR

    # fallback
    return A_DEBLUR


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
# MultiLoRA Param Collection (ONLY LoRA adapters, NOT base)
# --------------------------------------------------
def collect_multilora_params(toolbank: ToolBank, actions: List[str]) -> List[torch.nn.Parameter]:
    """
    Collect ONLY adapter params (A/B) for given actions from MultiLoRA wrappers.
    IMPORTANT: do NOT include base.* params.
    """
    params: List[torch.nn.Parameter] = []
    seen = set()

    for _, m in toolbank.backbone.named_modules():
        if isinstance(m, MultiLoRAConv2d):
            for a in actions:
                if a in m.lora_A:
                    for p in m.lora_A[a].parameters():
                        if id(p) not in seen:
                            p.requires_grad = True
                            params.append(p)
                            seen.add(id(p))
                if a in m.lora_B:
                    for p in m.lora_B[a].parameters():
                        if id(p) not in seen:
                            p.requires_grad = True
                            params.append(p)
                            seen.add(id(p))

        elif isinstance(m, MultiLoRALinear):
            for a in actions:
                if a in m.lora_A:
                    for p in m.lora_A[a].parameters():
                        if id(p) not in seen:
                            p.requires_grad = True
                            params.append(p)
                            seen.add(id(p))
                if a in m.lora_B:
                    for p in m.lora_B[a].parameters():
                        if id(p) not in seen:
                            p.requires_grad = True
                            params.append(p)
                            seen.add(id(p))

    return params


# --------------------------------------------------
# Action param cache (authoritative)
# --------------------------------------------------
def build_action_param_cache_multilora(
    toolbank: ToolBank, actions: List[str]
) -> Dict[str, List[torch.nn.Parameter]]:
    """
    cache[action] = list of LoRA adapter params (A/B) that belong to that action.
    This MUST be disjoint across actions for "independent params" goal.
    """
    cache: Dict[str, List[torch.nn.Parameter]] = {}
    for a in actions:
        cache[a] = []

    for _, m in toolbank.backbone.named_modules():
        if isinstance(m, MultiLoRAConv2d):
            for a in actions:
                if a in m.lora_A:
                    cache[a].extend(list(m.lora_A[a].parameters()))
                if a in m.lora_B:
                    cache[a].extend(list(m.lora_B[a].parameters()))

        elif isinstance(m, MultiLoRALinear):
            for a in actions:
                if a in m.lora_A:
                    cache[a].extend(list(m.lora_A[a].parameters()))
                if a in m.lora_B:
                    cache[a].extend(list(m.lora_B[a].parameters()))

    # dedup inside each action
    for a in actions:
        uniq, seen = [], set()
        for p in cache[a]:
            if id(p) not in seen:
                uniq.append(p)
                seen.add(id(p))
        cache[a] = uniq

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
# DIAG 0) optimizer param coverage check (ONCE)
# --------------------------------------------------
def diag_optimizer_coverage_once(
    toolbank: ToolBank,
    trainable_lora: List[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    action_param_cache: Dict[str, List[torch.nn.Parameter]],
    sample_k: int = 5,
):
    print("\n" + "-" * 78)
    print("[DIAG][OPT] optimizer param coverage check (ONCE)")

    # optimizer param ids
    opt_params = []
    for g in optimizer.param_groups:
        opt_params.extend(list(g["params"]))
    opt_ids = set(id(p) for p in opt_params)

    tl_ids = set(id(p) for p in trainable_lora)

    cache_all = []
    for _, ps in action_param_cache.items():
        cache_all.extend(ps)
    cache_ids = set(id(p) for p in cache_all)

    print(f"[DIAG][OPT] len(trainable_lora) = {len(trainable_lora)}")
    print(f"[DIAG][OPT] optimizer.param_groups = {len(optimizer.param_groups)}")
    print(f"[DIAG][OPT] optimizer total param count (sum of groups) = {len(opt_params)}")
    print(f"[DIAG][OPT] unique ids in optimizer = {len(opt_ids)}")
    print(f"[DIAG][OPT] unique ids in trainable_lora = {len(tl_ids)}")
    print(f"[DIAG][OPT] unique ids in action_param_cache(all) = {len(cache_ids)}")

    # set diffs
    not_in_opt = [p for p in trainable_lora if id(p) not in opt_ids]
    cache_not_in_opt = [p for p in cache_all if id(p) not in opt_ids]
    tl_not_in_cache = [p for p in trainable_lora if id(p) not in cache_ids]
    cache_not_in_tl = [p for p in cache_all if id(p) not in tl_ids]

    print(f"[DIAG][OPT] trainable_lora NOT in optimizer = {len(not_in_opt)}")
    print(f"[DIAG][OPT] action_param_cache params NOT in optimizer = {len(cache_not_in_opt)}")
    print(f"[DIAG][OPT] trainable_lora NOT in action_param_cache = {len(tl_not_in_cache)}")
    print(f"[DIAG][OPT] action_param_cache NOT in trainable_lora = {len(cache_not_in_tl)}")

    # build param -> name mapping (from named_parameters)
    name_of: Dict[int, str] = {}
    for n, p in toolbank.named_parameters():
        name_of[id(p)] = n

    # sample K
    k = min(sample_k, len(trainable_lora))
    idxs = random.sample(range(len(trainable_lora)), k=k) if k > 0 else []
    print(f"[DIAG][OPT] sample {k} params from trainable_lora:")
    for i in idxs:
        p = trainable_lora[i]
        nm = name_of.get(id(p), "<unnamed>")
        print(
            f"  - name={nm} id={id(p)} requires_grad={p.requires_grad} "
            f"dtype={p.dtype} device={p.device}"
        )

    # extra: ensure none of base weights are included
    base_like = [n for n, p in toolbank.named_parameters() if ("base." in n and id(p) in tl_ids)]
    if len(base_like) > 0:
        print("[DIAG][OPT][WARN] base.* params found inside trainable_lora (should be 0):")
        for n in base_like[:20]:
            print("  -", n)
    else:
        print("[DIAG][OPT] base.* params in trainable_lora = 0 (OK)")

    print("-" * 78 + "\n")


# --------------------------------------------------
# DIAG 1) scan current_action/scale/active
# --------------------------------------------------
def diag_scan_multilora_modules(backbone: nn.Module, limit: int = 8) -> str:
    lines = []
    cnt = 0
    for name, m in backbone.named_modules():
        if isinstance(m, (MultiLoRAConv2d, MultiLoRALinear)):
            cur = getattr(m, "current_action", None)
            s = float(m.scale.detach().cpu().item()) if hasattr(m, "scale") else None
            act = bool(getattr(m, "active", False))
            lines.append(f"[MultiLoRA] {name} :: current_action={cur} scale={s} active={act}")
            cnt += 1
            if cnt >= limit:
                break
    if cnt == 0:
        return "[MultiLoRA] No MultiLoRA wrappers found."
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


# --------------------------------------------------
# DIAG delta stats: mean/max/rms (vs STOP)
# --------------------------------------------------
@torch.no_grad()
def output_delta_stats_vs_stop(toolbank: ToolBank, x: torch.Tensor, action: str) -> Dict[str, float]:
    x1 = x[:1]
    out_stop = toolbank.apply(x1, A_STOP)
    out_act = toolbank.apply(x1, action)
    delta = (out_act - out_stop).float()
    mean_abs = float(delta.abs().mean().item())
    max_abs = float(delta.abs().max().item())
    rms = float(torch.sqrt(torch.mean(delta * delta)).item())
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
# Action helpers (NO HYBRID)
# --------------------------------------------------
def build_cycle_actions(cfg_tools: dict) -> List[str]:
    cyc = cfg_tools.get("train", {}).get("balance_cycle", None)
    if cyc is None:
        return [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE]
    # enforce no-hybrid
    out = []
    for a in list(cyc):
        if a == "A_HYBRID":
            continue
        out.append(a)
    return out


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

    diag_steps = cfg_tools.get("train", {}).get("diag_steps", [20, 40])
    if not isinstance(diag_steps, list) or len(diag_steps) == 0:
        diag_steps = [20, 40]
    diag_steps = [int(s) for s in diag_steps]
    diag_step_set = set(diag_steps)
    print(f"[Diag] diag_steps={diag_steps}")

    balance_cycle = build_cycle_actions(cfg_tools)
    actions_all = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE]

    lambda_silence = float(cfg_tools.get("train", {}).get("lambda_silence", 0.0))
    silence_warmup_steps = int(cfg_tools.get("train", {}).get("silence_warmup_steps", 0))
    silence_every = int(cfg_tools.get("train", {}).get("silence_every", 1))

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

    # toolbank specs (NO HYBRID)
    raw_specs = cfg_tools["toolbank"]["adapters"]
    adapter_specs: Dict[str, AdapterSpec] = {}
    for a in actions_all:
        s = raw_specs.get(a, {})
        adapter_specs[a] = AdapterSpec(**s)

    toolbank = ToolBank(
        backbone=backbone,
        adapter_specs=adapter_specs,
        device=device,
        debug=True,
    ).to(device)

    # freeze backbone base weights
    if bool(cfg_tools.get("toolbank", {}).get("freeze_backbone", True)):
        for p in toolbank.backbone.parameters():
            p.requires_grad = False

    # Trainables: ONLY MultiLoRA adapters for all actions
    trainable_lora = collect_multilora_params(toolbank, actions_all)
    lora_numel = sum(p.numel() for p in trainable_lora)
    print(f"[Trainable LoRA] {lora_numel/1e6:.2f} M params ({lora_numel} params)")

    optimizer = torch.optim.AdamW(
        [{"params": trainable_lora, "lr": lr, "weight_decay": 0.0}],
        betas=betas,
        weight_decay=0.0,
    )

    scaler = GradScaler("cuda", init_scale=scaler_init_scale, growth_interval=scaler_growth_interval) if use_amp else None

    # Action param cache + θ0 snapshot
    action_param_cache = build_action_param_cache_multilora(toolbank, actions_all)
    theta0_snapshot = build_action_theta0_snapshot_fp16(action_param_cache)
    print("[SilenceΔθ] snapshot(theta0) captured (fp16, cpu).")

    # OPT coverage check (ONCE)
    diag_optimizer_coverage_once(
        toolbank=toolbank,
        trainable_lora=trainable_lora,
        optimizer=optimizer,
        action_param_cache=action_param_cache,
        sample_k=5,
    )

    # DIAG pick params (per-action 1개씩)
    diag_picks = diag_pick_params(action_param_cache, actions_all, k_per_action=1)
    diag_snap: Dict[str, List[torch.Tensor]] = {}
    for a, ps in diag_picks.items():
        diag_snap[a] = [p.detach().clone().to("cpu", dtype=torch.float32) for p in ps]
    print("[DIAG] picked params per action:", {k: len(v) for k, v in diag_picks.items()})

    # Training cfg
    save_dir = cfg_tools["paths"]["ckpt_dir"]
    os.makedirs(save_dir, exist_ok=True)
    epochs = int(cfg_tools["train"]["epochs"])

    print(f"[Train] batch_size={cfg_ds['loader']['batch_size']} epochs={epochs} lr={lr} seed={cfg_ds.get('mixed', {}).get('seed', 123)}")
    print(f"[ActionBalance] cycle={balance_cycle}")
    print(f"[SilenceΔθ] lambda_silence={lambda_silence} warmup={silence_warmup_steps} every={silence_every}")
    print(f"[AMP] use_amp={use_amp} scaler_init_scale={scaler_init_scale} growth_interval={scaler_growth_interval}")
    print(f"[Sanity] sanity_every={sanity_every} diff_every={diff_every} log_every={log_every}")

    global_step = 0
    toolbank.train()

    action_loss_sum: DefaultDict[str, float] = defaultdict(float)
    action_count: DefaultDict[str, int] = defaultdict(int)
    action_seen: DefaultDict[str, int] = defaultdict(int)

    t0 = time.time()
    steps_per_epoch = len(loader)
    print(f"[Train] steps_per_epoch={steps_per_epoch} (MixedDataset.__len__)")

    cycle_idx = 0

    for epoch in range(epochs):
        total = 0.0
        action_loss_sum.clear()
        action_count.clear()

        pbar = tqdm(loader, ncols=120)

        for it, batch in enumerate(pbar, start=1):
            global_step += 1

            x = batch["input"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            oracle_actions = [choose_oracle_action_single(m["degradations"]) for m in batch["meta"]]
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
                torch.nn.utils.clip_grad_norm_(trainable_lora, grad_clip)

            # Decide diag/sanity/diff
            force_diag = (global_step in diag_step_set)
            do_sanity = (sanity_every > 0 and (global_step % sanity_every == 0))
            do_diff = (diff_every > 0 and (global_step % diff_every == 0))

            # DIAG 1) wrapper states BEFORE optimizer step
            if force_diag:
                print(f"[DIAG][multilora_state][pre_step] step={global_step} action={action}\n{diag_scan_multilora_modules(toolbank.backbone, limit=6)}")

            # SANITY grad stats
            if force_diag or do_sanity:
                tag = "DIAG" if force_diag else "SANITY"
                if use_amp:
                    try:
                        print(f"[{tag}][scaler] step={global_step} scale={scaler.get_scale():.1f}")
                    except Exception:
                        pass
                    ensure_unscale()

                # grad selectivity
                g_sel = grad_norm_for_action_params(action_param_cache, action)
                g_non = grad_norm_sum_except_params(action_param_cache, action, actions_all)
                ratio = g_sel / (g_non + 1e-12)

                # max grad
                max_g = 0.0
                cnt = 0
                for p in trainable_lora:
                    if p.grad is not None:
                        cnt += 1
                        max_g = max(max_g, float(p.grad.detach().abs().max().item()))

                print(f"[{tag}][grad_cnt] step={global_step} cnt={cnt}/{len(trainable_lora)}")
                print(f"[{tag}][grad_max] step={global_step} max_abs_grad={max_g:.3e}")
                print(
                    f"[{tag}][grad_sel] step={global_step} sel={action} "
                    f"g_sel={g_sel:.3e} sum_g_non_sel={g_non:.3e} ratio={ratio:.3e}"
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

            # DIAG/SANITY diff: Δout_vs_stop
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
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "act": action,
                    "m": f"{mismatch_rate:.2f}",
                    "min": f"{elapsed/60:.1f}",
                },
                refresh=False,
            )

            if global_step == 1 or (global_step % log_every == 0):
                tqdm.write(
                    f"[BATCH][debug] epoch={epoch+1}/{epochs} it={it}/{steps_per_epoch} step={global_step} "
                    f"target={action} oracle_major={oracle_major} mismatch={mismatch}/{len(oracle_actions)} "
                    f"ok={(oracle_major==action)} oracle_hist={dict(oracle_hist)}"
                )

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
        for a in actions_all:
            c = action_count.get(a, 0)
            if c > 0:
                m = action_loss_sum[a] / c
                print(f"  - {a:<8} : count={c:5d} mean_loss={m:.4f}")
            else:
                print(f"  - {a:<8} : count=    0 mean_loss=NA")

        print("[Seen so far]", {a: action_seen.get(a, 0) for a in actions_all})

    print("[Train] Done ✅")


if __name__ == "__main__":
    main()
