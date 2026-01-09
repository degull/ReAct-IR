# scripts/test_toolbank.py
import os
import sys
import glob

import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image

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
from models.backbone.vetnet import VETNet
from models.toolbank.toolbank import ToolBank, AdapterSpec
from models.planner.action_space import (
    A_DEDROP, A_DEBLUR, A_DERAIN, A_DESNOW, A_DEHAZE, A_HYBRID
)

import yaml

# --------------------------------------------------
# Config
# --------------------------------------------------
CKPT_PATH = "E:/ReAct-IR/checkpoints/toolbank/epoch_001_loss0.1042.pth"

INPUT_SINGLE = "E:/ReAct-IR/test_data/raindrop/00001.png"
INPUT_FOLDER = "E:/ReAct-IR/test_images/folder"

OUT_ROOT = "E:/ReAct-IR/results/toolbank_test"

ACTIONS = [
    A_DEHAZE,
    A_DERAIN,
    A_DEDROP,
    A_DESNOW,
    A_DEBLUR,
    A_HYBRID,
]

# --------------------------------------------------
# Utils
# --------------------------------------------------
def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    t = torch.from_numpy(arr.transpose(2, 0, 1))
    return t.unsqueeze(0)

def load_image(path):
    return Image.open(path).convert("RGB")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # ---------------------------
    # Load configs
    # ---------------------------
    cfg_tools = load_yaml(os.path.join(PROJECT_ROOT, "configs", "tools.yaml"))

    # ---------------------------
    # Build model
    # ---------------------------
    backbone = VETNet(dim=48, volterra_rank=4).to(device)

    adapter_specs = {
        a: AdapterSpec(**s)
        for a, s in cfg_tools["toolbank"]["adapters"].items()
    }

    toolbank = ToolBank(
        backbone,
        adapter_specs,
        device=device,
        debug=False
    ).to(device)

    # ---------------------------
    # Load checkpoint
    # ---------------------------
    print(f"[Load] {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    toolbank.load_state_dict(ckpt["toolbank"], strict=True)

    toolbank.eval()

    # ==================================================
    # 1️⃣ Single Image Test
    # ==================================================
    os.makedirs(os.path.join(OUT_ROOT, "single"), exist_ok=True)

    img = load_image(INPUT_SINGLE)
    x = pil_to_tensor(img).to(device)

    save_image(x, os.path.join(OUT_ROOT, "single", "input.png"))

    with torch.no_grad():
        for act in ACTIONS:
            print(f"[Single] Action = {act}")
            out = toolbank.apply(x, act).clamp(0, 1)
            save_image(
                out,
                os.path.join(OUT_ROOT, "single", f"{act}.png")
            )

    # ==================================================
    # 2️⃣ Folder Test
    # ==================================================
    img_paths = sorted(
        glob.glob(os.path.join(INPUT_FOLDER, "*.*"))
    )

    for p in img_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        out_dir = os.path.join(OUT_ROOT, "folder", name)
        os.makedirs(out_dir, exist_ok=True)

        img = load_image(p)
        x = pil_to_tensor(img).to(device)
        save_image(x, os.path.join(out_dir, "input.png"))

        with torch.no_grad():
            for act in ACTIONS:
                print(f"[Folder:{name}] Action = {act}")
                out = toolbank.apply(x, act).clamp(0, 1)
                save_image(
                    out,
                    os.path.join(out_dir, f"{act}.png")
                )

    print("[Test] Done ✅")

if __name__ == "__main__":
    main()
