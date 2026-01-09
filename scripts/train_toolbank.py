# scripts/train_toolbank.py
import os
import sys
import time
import random
from typing import List, Dict, DefaultDict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
def choose_oracle_action(degs: List[str]) -> str:
    d = set(x.lower() for x in degs)
    if "drop" in d:
        return A_DEDROP
    if "rain" in d:
        return A_DERAIN
    if "haze" in d:
        return A_DEHAZE
    if "snow" in d:
        return A_DESNOW
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
# ToolBank Sanity Metrics (핵심: 학습 중 on/off 검증)
# --------------------------------------------------
def lora_grad_norm_for_action(toolbank: ToolBank, action: str) -> float:
    """
    현재 step에서 action에 해당하는 LoRA 모듈들의 grad L2 norm 합.
    - action이 제대로 분기되면: 선택된 action의 grad norm이 상대적으로 커지고,
      선택 안 된 action들은 매우 작아야 정상.
    """
    total = 0.0
    modules = toolbank.adapters.get(action, [])
    for m in modules:
        for p in m.parameters():
            if p.grad is not None:
                total += float(p.grad.data.norm(2).item())
    return total


@torch.no_grad()
def output_diff_vs_stop(toolbank: ToolBank, x: torch.Tensor, action: str) -> float:
    """
    같은 입력 x에 대해:
      - LoRA OFF (A_STOP) 출력
      - 해당 action LoRA ON 출력
    두 출력의 평균 절대차를 리턴.
    - 0에 가깝다면 LoRA가 사실상 영향이 없는 상태일 수 있음.
    """
    x1 = x[:1]  # 비용 절감: 1개 샘플만
    toolbank.activate_adapter(A_STOP)
    out_stop = toolbank.backbone(x1)

    toolbank.activate_adapter(action)
    out_act = toolbank.backbone(x1)

    return float((out_act - out_stop).abs().mean().item())


@torch.no_grad()
def action_separation_score(
    toolbank: ToolBank,
    x: torch.Tensor,
    sel_action: str,
    neg_action: str,
) -> float:
    """
    (추가 SANITY) action separation score:
      score = mean(|f_sel(x) - f_neg(x)|) / (mean(|f_sel(x) - f_stop(x)|) + eps)

    - 분리(score ↑): 선택 action 출력이 다른 action과 충분히 다름
    - 정상화(denom): LoRA 자체의 유효 영향(Stop 대비)을 기준으로 스케일 불변 비교
    """
    eps = 1e-8
    x1 = x[:1]

    toolbank.activate_adapter(A_STOP)
    out_stop = toolbank.backbone(x1)

    toolbank.activate_adapter(sel_action)
    out_sel = toolbank.backbone(x1)

    toolbank.activate_adapter(neg_action)
    out_neg = toolbank.backbone(x1)

    num = float((out_sel - out_neg).abs().mean().item())
    den = float((out_sel - out_stop).abs().mean().item())
    return float(num / (den + eps))


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
# Action-Contrastive helpers (최소 변경)
# --------------------------------------------------
ALL_ACTIONS = [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID]


def sample_negative_action(pos_action: str, oracle_actions: List[str]) -> str:
    """
    negative action 1개 샘플.
    - 기본: pos_action 제외한 전체 action 중 랜덤
    - oracle_actions(배치 내 다양성)가 있다면 거기서 먼저 뽑아주는 게 더 'hard negative'가 될 수 있음
    """
    # hard negative 우선: 같은 배치 oracle 안에서 pos와 다른 action이 있으면 거기서 선택
    uniq = list(set(oracle_actions))
    cand = [a for a in uniq if a != pos_action]
    if len(cand) > 0:
        return random.choice(cand)

    # fallback: 전체 action 중 pos 제외
    cand2 = [a for a in ALL_ACTIONS if a != pos_action]
    return random.choice(cand2)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    cfg_ds = load_yaml(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    cfg_tools = load_yaml(os.path.join(PROJECT_ROOT, "configs", "tools.yaml"))

    set_seed(int(cfg_ds.get("mixed", {}).get("seed", 123)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # ------------------------------
    # Sanity logging frequency
    # ------------------------------
    # tools.yaml에 없으면 기본값 사용
    sanity_every = int(cfg_tools.get("train", {}).get("sanity_every", 200))
    diff_every = int(cfg_tools.get("train", {}).get("diff_every", 500))

    # ------------------------------
    # (Condition-1) Action 실패 시 손해: action_specific_penalty
    #   - 배치 내 oracle action과 선택 action이 불일치한 비율(mismatch_rate)에 비례해 loss에 가산
    # ------------------------------
    action_specific_penalty = float(cfg_tools.get("train", {}).get("action_specific_penalty", 0.0))

    # ------------------------------
    # (Condition-3) 되돌릴 수 없는 선택: switch_cost
    #   - 직전 step의 action과 다르면 loss에 switch_cost 가산
    # ------------------------------
    switch_cost = float(cfg_tools.get("train", {}).get("switch_cost", 0.0))

    # ------------------------------
    # (NEW) Action-Contrastive Loss
    #   loss = L_rec + lambda_contrast * |f_a(x) - f_a'(x)|_1
    #   - a'는 negative action 1개만 샘플
    # ------------------------------
    lambda_contrast = float(cfg_tools.get("train", {}).get("lambda_contrast", 0.0))
    contrast_warmup_steps = int(cfg_tools.get("train", {}).get("contrast_warmup_steps", 0))
    contrast_every = int(cfg_tools.get("train", {}).get("contrast_every", 1))  # 매 step 계산하면 비용↑, N step마다 적용 가능

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

    # Training
    save_dir = cfg_tools["paths"]["ckpt_dir"]
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(cfg_tools["train"]["epochs"])
    save_every = int(cfg_tools["train"].get("save_every", 2000))

    global_step = 0
    toolbank.train()

    # epoch별 action 통계/손실 누적
    action_loss_sum: DefaultDict[str, float] = defaultdict(float)
    action_count: DefaultDict[str, int] = defaultdict(int)

    # (Condition-3) 직전 action tracking
    prev_action: str = A_STOP

    t0 = time.time()

    for epoch in range(epochs):
        total = 0.0
        action_loss_sum.clear()
        action_count.clear()

        pbar = tqdm(loader, ncols=120)

        for batch in pbar:
            global_step += 1

            x = batch["input"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            # 배치 각 샘플의 oracle action
            oracle_actions = [choose_oracle_action(m["degradations"]) for m in batch["meta"]]
            # 현재 step에서 사용할 action (다수결)
            action = max(set(oracle_actions), key=oracle_actions.count)

            # (Condition-1) action 실패 손해: mismatch rate 기반 penalty
            if action_specific_penalty > 0.0:
                mismatch = sum(1 for a in oracle_actions if a != action)
                mismatch_rate = float(mismatch) / float(max(1, len(oracle_actions)))
                loss_pen_action = action_specific_penalty * mismatch_rate
            else:
                mismatch_rate = 0.0
                loss_pen_action = 0.0

            # (Condition-3) switch cost: 되돌릴 수 없는 선택(전환 비용)
            if switch_cost > 0.0 and prev_action != A_STOP and action != prev_action:
                loss_pen_switch = switch_cost
            else:
                loss_pen_switch = 0.0

            # (NEW) negative action 샘플 (contrast 계산 시 필요)
            neg_action = None
            use_contrast = (lambda_contrast > 0.0)
            if use_contrast:
                if global_step >= max(0, contrast_warmup_steps) and (contrast_every <= 1 or (global_step % contrast_every == 0)):
                    neg_action = sample_negative_action(action, oracle_actions)
                else:
                    use_contrast = False  # warmup/every 조건 미충족이면 이번 step은 contrast off

            with autocast(device_type="cuda"):
                # main prediction
                pred = toolbank.apply(x, action)
                loss_rec = charbonnier_loss(pred, gt)

                # 최종 loss = 복원 loss + (조건1) + (조건3) + (NEW contrast)
                loss = loss_rec

                if loss_pen_action > 0.0:
                    loss = loss + loss_rec.new_tensor(loss_pen_action)
                if loss_pen_switch > 0.0:
                    loss = loss + loss_rec.new_tensor(loss_pen_switch)

                # (NEW) Action-Contrastive Loss (negative 1개)
                # - 같은 x로 다른 action 출력도 계산 (추가 forward 1회)
                loss_contrast_val = 0.0
                if use_contrast and (neg_action is not None):
                    # contrast는 1 sample만
                    x_c = x[:1]
                    pred_c = pred[:1]
                    pred_neg = toolbank.apply(x_c, neg_action)
                    l_con = torch.mean(torch.abs(pred_c - pred_neg))
                    loss = loss + (loss_rec.new_tensor(lambda_contrast) * l_con)
                    loss_contrast_val = float(l_con.detach().item())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # ------------------------------
            # SANITY 1) action별 grad norm (unscaled grad로 측정)
            # ------------------------------
            if sanity_every > 0 and (global_step % sanity_every == 0):
                # unscale to read true grad magnitudes
                scaler.unscale_(optimizer)

                g_sel = lora_grad_norm_for_action(toolbank, action)
                g_hyb = lora_grad_norm_for_action(toolbank, A_HYBRID) if action != A_HYBRID else 0.0
                g_drop = lora_grad_norm_for_action(toolbank, A_DEDROP) if action != A_DEDROP else 0.0
                g_blur = lora_grad_norm_for_action(toolbank, A_DEBLUR) if action != A_DEBLUR else 0.0

                print(
                    f"[SANITY][grad] step={global_step} act={action} "
                    f"g_sel={g_sel:.4f} g_hyb={g_hyb:.4f} g_drop={g_drop:.4f} g_blur={g_blur:.4f}"
                )

            # optimizer step
            scaler.step(optimizer)
            scaler.update()

            # ------------------------------
            # SANITY 2) 출력 차이(Stop 대비) - 가끔만
            # ------------------------------
            sep_score = None
            if diff_every > 0 and (global_step % diff_every == 0):
                toolbank.eval()
                with torch.no_grad():
                    d_out = output_diff_vs_stop(toolbank, x, action)

                    # (NEW) action separation score
                    # - 이번 step에 neg_action이 없었으면(contrast off) 하드네거티브 하나 다시 샘플
                    neg_for_sanity = neg_action if (neg_action is not None) else sample_negative_action(action, oracle_actions)
                    sep_score = action_separation_score(toolbank, x, action, neg_for_sanity)

                toolbank.train()
                print(f"[SANITY][diff] step={global_step} act={action} Δout_vs_stop={d_out:.6f}")
                print(f"[SANITY][sep ] step={global_step} act={action} neg={neg_for_sanity} sep_score={sep_score:.4f}")

            # loss/stat updates
            loss_val = float(loss.item())
            total += loss_val
            avg_loss = total / (pbar.n + 1)

            action_loss_sum[action] += loss_val
            action_count[action] += 1

            elapsed = time.time() - t0

            # progress text (기존 출력 유지 + penalty/contrast 정보만 덧붙임)
            # - 기존: loss, act, m, sw, t 유지
            # - 추가: con(contrast L1)만 표시 (계산 안 한 step은 0.000)
            sw_flag = 1 if (prev_action != A_STOP and action != prev_action) else 0
            pbar.set_description(
                f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | act={action} | "
                f"m={mismatch_rate:.2f} | sw={sw_flag} | con={loss_contrast_val:.4f} | "
                f"t={elapsed/60:.1f}m"
            )

            # Iteration checkpoint
            if global_step % save_every == 0:
                iter_path = os.path.join(save_dir, f"iter_{global_step}_loss{avg_loss:.4f}.pth")
                torch.save({"toolbank": toolbank.state_dict(), "step": global_step}, iter_path)
                print(f"[CKPT] Saved: {iter_path}")

            # (Condition-3) prev_action 갱신 (step 끝에서 업데이트)
            prev_action = action

        # Epoch checkpoint
        epoch_loss = total / max(1, len(loader))
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

        # ------------------------------
        # Epoch summary: action-wise loss
        # ------------------------------
        print(f"[Epoch {epoch+1}] Action-wise mean loss:")
        for a in [A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID]:
            c = action_count.get(a, 0)
            if c > 0:
                m = action_loss_sum[a] / c
                print(f"  - {a:<8} : count={c:5d} mean_loss={m:.4f}")
            else:
                print(f"  - {a:<8} : count=    0 mean_loss=NA")

    print("[Train] Done ✅")


if __name__ == "__main__":
    main()
