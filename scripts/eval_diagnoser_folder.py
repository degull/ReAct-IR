import os
import argparse
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.amp import autocast

# ----------------------------
# Label / Action mapping
# ----------------------------
LABEL_NAMES = ["blur", "rain", "snow", "haze", "drop"]
IDX_TO_ACTION = {
    0: "A_DEBLUR",
    1: "A_DERAIN",
    2: "A_DESNOW",
    3: "A_DEHAZE",
    4: "A_DEDROP",
}
ACTION_TO_IDX = {v: k for k, v in IDX_TO_ACTION.items()}

# GT inference from folder name
# (you can extend this mapping any time)
FOLDER_TO_ACTION = {
    "blur": "A_DEBLUR",
    "rain": "A_DERAIN",
    "snow": "A_DESNOW",
    "hazy": "A_DEHAZE",
    "haze": "A_DEHAZE",
    "drop": "A_DEDROP",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------
# Model definition (must match train)
# ----------------------------
class TinyViTDiagnoser(nn.Module):
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
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

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
        self.head = nn.Linear(embed_dim, num_labels)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                 # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)        # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, x], dim=1)          # (B,1+N,D)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)                     # (B,5)

def load_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)

def load_img_tensor(path: str, patch: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    W, H = img.size

    if H >= patch and W >= patch:
        x0 = (W - patch) // 2
        y0 = (H - patch) // 2
        img = img.crop((x0, y0, x0 + patch, y0 + patch))
    else:
        img = img.resize((patch, patch), resample=Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x

def infer_gt_action_from_path(path: str) -> str:
    # Use any folder component that matches our mapping
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    for p in reversed(parts):  # closer folder wins
        if p in FOLDER_TO_ACTION:
            return FOLDER_TO_ACTION[p]
    return "UNKNOWN"

def iter_images(root: str):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                yield os.path.join(dp, fn)

def print_confusion(cm: np.ndarray):
    names = [IDX_TO_ACTION[i] for i in range(5)]
    w = max(len(n) for n in names)
    header = "GT\\PR".ljust(w) + "  " + "  ".join([n.rjust(w) for n in names])
    print("\n[Confusion Matrix] (rows=GT, cols=Pred)")
    print(header)
    for i, gt_name in enumerate(names):
        row = cm[i]
        print(gt_name.ljust(w) + "  " + "  ".join([str(int(v)).rjust(w) for v in row]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="E:/ReAct-IR/checkpoints/diagnoser/diagnoser_best.pth")
    ap.add_argument("--root", required=True, help="folder to evaluate (GT inferred from folder names)")
    ap.add_argument("--out_csv", default="E:/ReAct-IR/results/diagnoser_eval/diagnoser_eval.csv")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--max_items", type=int, default=-1, help="-1 = all")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[CKPT]", args.ckpt)
    print("[Root]", args.root)

    model = TinyViTDiagnoser(img_size=args.patch, patch_size=16, embed_dim=384, depth=6, num_heads=6, num_labels=5)
    load_ckpt(model, args.ckpt)
    model.to(device).eval()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    cm = np.zeros((5, 5), dtype=np.int64)
    total = 0
    correct = 0
    unknown = 0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "gt_action", "pred_action"] + [f"s0_{n}" for n in LABEL_NAMES])

        for i, path in enumerate(iter_images(args.root)):
            if args.max_items > 0 and i >= args.max_items:
                break

            gt_action = infer_gt_action_from_path(path)
            if gt_action == "UNKNOWN" or gt_action not in ACTION_TO_IDX:
                unknown += 1
                continue

            x = load_img_tensor(path, patch=args.patch).to(device)

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=(bool(args.use_amp) and device.type == "cuda")):
                    logits = model(x)
                probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_action = IDX_TO_ACTION[pred_idx]

            gt_idx = ACTION_TO_IDX[gt_action]
            cm[gt_idx, pred_idx] += 1

            total += 1
            if pred_action == gt_action:
                correct += 1

            wr.writerow([path, gt_action, pred_action] + [f"{float(probs[k]):.6f}" for k in range(5)])

    acc = (correct / total) if total > 0 else 0.0
    print(f"\n[Done] total_used={total} correct={correct} acc={acc:.4f} skipped_unknown={unknown}")
    print_confusion(cm)

    # per-class accuracy
    print("\n[Per-class Acc]")
    for i in range(5):
        gt_name = IDX_TO_ACTION[i]
        denom = cm[i].sum()
        a = (cm[i, i] / denom) if denom > 0 else 0.0
        print(f"  {gt_name}: {a:.4f}  (n={int(denom)})")

    print(f"\n[CSV] saved -> {args.out_csv}")

if __name__ == "__main__":
    main()
