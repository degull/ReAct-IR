# datasets/reside6k.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from typing import List, Tuple

from datasets.base_dataset import BaseDataset, DatasetDebugConfig


class RESIDE6KDataset(BaseDataset):
    """
    RESIDE-6K Dataset Loader (robust)

    Common layouts supported:

    RESIDE-6K/
      ├─ train/ or Train/
      │   ├─ hazy/ or Hazy/
      │   └─ clear/ or Clear/
      └─ test/ or Test/
          ├─ hazy/
          └─ clear/

    Some variants:
      - input / gt
      - hazy / gt
    We search common candidates.
    """

    degradations = ["haze", "blur", "noise"]

    def _find_existing(self, base: str, candidates: List[str]) -> str:
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                return p
        raise RuntimeError(
            f"[RESIDE6KDataset] None of candidate folders exist under:\n"
            f"  base={base}\n"
            f"  candidates={candidates}\n"
            f"  -> Please check your RESIDE-6K folder structure."
        )

    def _resolve_split_dir(self, base_root: str) -> Tuple[str, bool]:
        for c in [self.split, self.split.capitalize(), self.split.upper()]:
            p = os.path.join(base_root, c)
            if os.path.isdir(p):
                return p, True
        return base_root, False

    def _load_paths(self):
        ds_root = os.path.join(self.root, "RESIDE-6K")
        if not os.path.isdir(ds_root):
            raise RuntimeError(f"[RESIDE6KDataset] Root not found: {ds_root}")

        split_root, has_split = self._resolve_split_dir(ds_root)

        if self.debug.enabled:
            print(f"[RESIDE6KDataset] base={ds_root}")
            print(f"[RESIDE6KDataset] resolved split_root={split_root} (has_split={has_split})")

        inp_dir = self._find_existing(
            split_root,
            ["hazy", "Hazy", "input", "Input", "images", "Images"]
        )
        gt_dir = self._find_existing(
            split_root,
            ["clear", "Clear", "gt", "GT", "clean", "Clean", "target", "Target"]
        )

        # Pair by filename
        # NOTE: In RESIDE, hazy filenames sometimes contain extra suffix.
        # If your dataset uses different naming (e.g., xxx_0.8.png),
        # we will need a custom pairing rule. For now we do strict basename match.
        inps, gts, stats = self.pair_by_filename(inp_dir=inp_dir, gt_dir=gt_dir, recursive=False)

        if len(inps) == 0:
            raise RuntimeError(
                "[RESIDE6KDataset] No paired samples found with basename matching.\n"
                f"  inp_dir={inp_dir}\n"
                f"  gt_dir={gt_dir}\n"
                "  If your hazy filenames include suffix (e.g., *_0.9.png), "
                "we need custom pairing logic (stem match)."
            )

        self.input_paths = inps
        self.gt_paths = gts


# ------------------------------
# Debug main
# ------------------------------
if __name__ == "__main__":
    """
    Usage:
      python datasets/reside6k.py

    Expected:
      E:/ReAct-IR/data/RESIDE-6K/...
    """
    ROOT = "E:/ReAct-IR/data"

    debug_cfg = DatasetDebugConfig(
        enabled=True,
        verbose=True,
        show_first_k=3,
        strict_pairing=True,
        skip_missing_pairs=True,
    )

    print("\n[DEBUG] Initializing RESIDE6KDataset...")
    ds = RESIDE6KDataset(
        root=ROOT,
        split="train",
        transform=None,
        debug=debug_cfg
    )
    print(f"\n[DEBUG] Dataset length: {len(ds)}")

    sample = ds[0]
    print("\n[DEBUG] Sample meta:")
    for k, v in sample["meta"].items():
        print(f"  {k}: {v}")

    print("[DEBUG] RESIDE6KDataset OK ✅")
