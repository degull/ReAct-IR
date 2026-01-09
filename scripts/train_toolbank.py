""" # scripts/train_toolbank.py
import os
import sys
import time
import random
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
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
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DENOISE, A_DEHAZE, A_DEJPEG, A_HYBRID
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
        gt  = gt.resize((nw, nh), Image.BILINEAR)
        w, h = inp.size
    x = random.randint(0, w - patch)
    y = random.randint(0, h - patch)
    return inp.crop((x, y, x + patch, y + patch)), gt.crop((x, y, x + patch, y + patch))

def _augment_pair(inp, gt):
    if random.random() < 0.5:
        inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
        gt  = gt.transpose(Image.FLIP_LEFT_RIGHT)
    k = random.randint(0, 3)
    if k > 0:
        inp = inp.rotate(90 * k, expand=True)
        gt  = gt.rotate(90 * k, expand=True)
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
def choose_oracle_action(degs: List[str]) -> str:
    d = set(x.lower() for x in degs)
    if "drop" in d:  return A_DEDROP
    if "rain" in d:  return A_DERAIN
    if "haze" in d:  return A_DEHAZE
    if "noise" in d: return A_DENOISE
    if "jpeg" in d:  return A_DEJPEG
    if "blur" in d:  return A_DEBLUR
    return A_HYBRID

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
    for m in model.modules():
        if hasattr(m, "is_lora") and m.is_lora:
            for p in m.parameters():
                if id(p) not in seen:
                    p.requires_grad = True
                    params.append(p)
                    seen.add(id(p))
    return params

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
# Main
# --------------------------------------------------
def main():
    cfg_ds = load_yaml(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    cfg_tools = load_yaml(os.path.join(PROJECT_ROOT, "configs", "tools.yaml"))

    set_seed(int(cfg_ds.get("mixed", {}).get("seed", 123)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

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
    toolbank = ToolBank(backbone, adapter_specs, device=device, debug=True).to(device)

    for p in toolbank.backbone.parameters():
        p.requires_grad = False

    trainable = collect_lora_params(toolbank)
    print(f"[Trainable LoRA] {sum(p.numel() for p in trainable)/1e6:.2f} M params")

    optimizer = torch.optim.AdamW(trainable, lr=cfg_tools["train"]["lr"])
    scaler = GradScaler("cuda")

    # Training
    save_dir = cfg_tools["paths"]["ckpt_dir"]
    os.makedirs(save_dir, exist_ok=True)

    epochs = cfg_tools["train"]["epochs"]
    save_every = cfg_tools["train"].get("save_every", 2000)

    global_step = 0
    toolbank.train()

    for epoch in range(epochs):
        total = 0.0
        pbar = tqdm(loader, ncols=120)

        for batch in pbar:
            global_step += 1

            x = batch["input"].to(device)
            gt = batch["gt"].to(device)

            actions = [choose_oracle_action(m["degradations"]) for m in batch["meta"]]
            action = max(set(actions), key=actions.count)

            with autocast(device_type="cuda"):
                pred = toolbank.apply(x, action)
                loss = charbonnier_loss(pred, gt)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += loss.item()
            avg_loss = total / (pbar.n + 1)

            pbar.set_description(
                f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | act={action}"
            )

            # Iteration checkpoint (loss 포함)
            if global_step % save_every == 0:
                iter_path = os.path.join(
                    save_dir, f"iter_{global_step}_loss{avg_loss:.4f}.pth"
                )
                torch.save({"toolbank": toolbank.state_dict()}, iter_path)

        # Epoch checkpoint (loss 포함)
        epoch_loss = total / len(loader)
        ckpt_path = os.path.join(
            save_dir, f"epoch_{epoch+1:03d}_loss{epoch_loss:.4f}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "toolbank": toolbank.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"[CKPT] Saved: {ckpt_path}")

    print("[Train] Done ✅")

if __name__ == "__main__":
    main()
 """

# 이어서 학습
# scripts/train_toolbank.py
# scripts/train_toolbank.py
import os
import sys
import random
from typing import List

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
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DENOISE, A_DEHAZE, A_DEJPEG, A_HYBRID
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
    return (
        inp.crop((x, y, x + patch, y + patch)),
        gt.crop((x, y, x + patch, y + patch)),
    )


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
def choose_oracle_action(degs: List[str]) -> str:
    d = set(x.lower() for x in degs)
    if "drop" in d:
        return A_DEDROP
    if "rain" in d:
        return A_DERAIN
    if "haze" in d:
        return A_DEHAZE
    if "noise" in d:
        return A_DENOISE
    if "jpeg" in d:
        return A_DEJPEG
    if "blur" in d:
        return A_DEBLUR
    return A_HYBRID


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
# LoRA Param Collection
# --------------------------------------------------
def collect_lora_params(model: nn.Module):
    params = []
    seen = set()
    for m in model.modules():
        if hasattr(m, "is_lora") and m.is_lora:
            for p in m.parameters():
                if id(p) not in seen:
                    p.requires_grad = True
                    params.append(p)
                    seen.add(id(p))
    return params


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
# Main
# --------------------------------------------------
def main():
    cfg_ds = load_yaml(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    cfg_tools = load_yaml(os.path.join(PROJECT_ROOT, "configs", "tools.yaml"))

    set_seed(int(cfg_ds.get("mixed", {}).get("seed", 123)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

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
    toolbank = ToolBank(backbone, adapter_specs, device=device, debug=True).to(device)

    # Freeze backbone (LoRA만 학습)
    for p in toolbank.backbone.parameters():
        p.requires_grad = False

    trainable = collect_lora_params(toolbank)
    print(f"[Trainable LoRA] {sum(p.numel() for p in trainable)/1e6:.2f} M params")

    optimizer = torch.optim.AdamW(trainable, lr=cfg_tools["train"]["lr"])
    scaler = GradScaler("cuda")

    # Save dir
    save_dir = cfg_tools["paths"]["ckpt_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Training configs
    epochs = int(cfg_tools["train"]["epochs"])
    save_every = 250  # 250 iteration마다 저장

    # --------------------------------------------------
    # ✅ Resume (epoch ckpt 기준)
    #   - start_epoch = ckpt["epoch"]
    #   - global_step = start_epoch * len(loader)
    # --------------------------------------------------
    resume_path = cfg_tools.get("train", {}).get("resume_path", None)
    start_epoch = 0
    global_step = 0

    if resume_path is not None and str(resume_path).strip() != "":
        print(f"[RESUME] Loading {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        # 필수: toolbank
        toolbank.load_state_dict(ckpt["toolbank"], strict=True)

        # 선택: optimizer
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        # ✅ 핵심 수정 포인트
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = start_epoch * len(loader)

        print(f"[RESUME] start_epoch={start_epoch}, global_step={global_step}")
    else:
        print("[RESUME] None (start from scratch)")

    toolbank.train()

    # --------------------------------------------------
    # Training Loop (✅ for epoch in range(start_epoch, epochs))
    # --------------------------------------------------
    for epoch in range(start_epoch, epochs):
        total = 0.0
        pbar = tqdm(loader, ncols=120)

        for batch in pbar:
            global_step += 1

            x = batch["input"].to(device)
            gt = batch["gt"].to(device)

            actions = [choose_oracle_action(m["degradations"]) for m in batch["meta"]]
            action = max(set(actions), key=actions.count)

            with autocast(device_type="cuda"):
                pred = toolbank.apply(x, action)
                loss = charbonnier_loss(pred, gt)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += loss.item()
            avg_loss = total / (pbar.n + 1)

            pbar.set_description(
                f"Epoch {epoch+1}/{epochs} | step={global_step} | loss={avg_loss:.4f} | act={action}"
            )

            # Iteration checkpoint (every 250)
            if global_step % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"iter_{global_step}_loss{avg_loss:.4f}.pth")
                torch.save(
                    {
                        "epoch": epoch,  # 현재 epoch index (0-based)
                        "toolbank": toolbank.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                print(f"\n[CKPT] Saved: {ckpt_path}")

        # Epoch checkpoint (epoch 끝날 때마다 저장하고 싶으면 주석 해제)
        epoch_loss = total / max(1, len(loader))
        epoch_ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}_loss{epoch_loss:.4f}.pth")
        torch.save(
            {
                "epoch": epoch + 1,  # ✅ 다음 epoch 시작값으로 쓰기 위해 +1 저장
                "toolbank": toolbank.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            },
            epoch_ckpt_path,
        )
        print(f"[CKPT] Saved: {epoch_ckpt_path}")

    print("[Train] Done ✅")


if __name__ == "__main__":
    main()
