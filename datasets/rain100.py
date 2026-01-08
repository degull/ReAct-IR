# datasets/rain100.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import List, Tuple

from datasets.base_dataset import BaseDataset, DatasetDebugConfig



class Rain100Dataset(BaseDataset):
    """
    Robust Rain100 Dataset Loader
    Supports multiple directory layouts automatically.

    Supported layouts:

    (A)
    rain100H/
      ├─ train/
      │   ├─ rain/
      │   └─ norain/
      └─ test/
          ├─ rain/
          └─ norain/

    (B)
    rain100H/
      ├─ rain/
      └─ norain/
    """

    degradations = ["rain", "blur"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        heavy: bool = True,
        transform=None,
        debug: DatasetDebugConfig | None = None,
    ):
        self.heavy = heavy
        super().__init__(root, split, transform, debug)

    # ------------------------------
    # Utilities
    # ------------------------------
    def _find_existing(self, base: str, candidates: List[str]) -> str:
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                return p
        raise RuntimeError(
            f"[Rain100Dataset] None of the candidate folders exist under:\n"
            f"  base={base}\n"
            f"  candidates={candidates}"
        )

    def _resolve_split_root(self, base_root: str) -> Tuple[str, bool]:
        """
        Decide whether split subfolder exists.
        Returns:
          (resolved_root, has_split)
        """
        for c in [self.split, self.split.capitalize(), self.split.upper()]:
            p = os.path.join(base_root, c)
            if os.path.isdir(p):
                return p, True
        return base_root, False

    # ------------------------------
    # Core loader
    # ------------------------------
    def _load_paths(self):
        name = "rain100H" if self.heavy else "rain100L"
        base_root = os.path.join(self.root, name)

        if not os.path.isdir(base_root):
            raise RuntimeError(f"[Rain100Dataset] Root not found: {base_root}")

        # 1) resolve split root (train/test or flat)
        data_root, has_split = self._resolve_split_root(base_root)

        if self.debug.enabled:
            print(f"[Rain100Dataset] Using base: {base_root}")
            print(f"[Rain100Dataset] Resolved data root: {data_root}")
            print(f"[Rain100Dataset] Split folder detected: {has_split}")

        # 2) find rain / norain directories
        inp_dir = self._find_existing(
            data_root,
            ["rain", "Rain", "RAIN", "input", "Input"]
        )
        gt_dir = self._find_existing(
            data_root,
            ["norain", "NoRain", "NORAIN", "gt", "GT", "groundtruth"]
        )

        # 3) pair images
        inps, gts, stats = self.pair_by_filename(
            inp_dir=inp_dir,
            gt_dir=gt_dir,
            recursive=False
        )

        if len(inps) == 0:
            raise RuntimeError(
                "[Rain100Dataset] No paired samples found.\n"
                f"  inp_dir={inp_dir}\n"
                f"  gt_dir={gt_dir}"
            )

        self.input_paths = inps
        self.gt_paths = gts


# ------------------------------
# Debug main
# ------------------------------
if __name__ == "__main__":
    """
    Usage:
      python datasets/rain100.py

    Expected data location:
      E:/ReAct-IR/data/rain100H
      E:/ReAct-IR/data/rain100L
    """

    ROOT = "E:/ReAct-IR/data"

    debug_cfg = DatasetDebugConfig(
        enabled=True,
        verbose=True,
        show_first_k=3,
        strict_pairing=True,
        skip_missing_pairs=True,
    )

    print("\n[DEBUG] Initializing Rain100H Dataset...")
    ds_h = Rain100Dataset(
        root=ROOT,
        split="train",
        heavy=True,
        transform=None,
        debug=debug_cfg,
    )
    print(f"[DEBUG] Rain100H length: {len(ds_h)}")

    print("\n[DEBUG] Initializing Rain100L Dataset...")
    ds_l = Rain100Dataset(
        root=ROOT,
        split="test",
        heavy=False,
        transform=None,
        debug=debug_cfg,
    )
    print(f"[DEBUG] Rain100L length: {len(ds_l)}")

    sample = ds_h[0]
    print("\n[DEBUG] Sample meta (Rain100H):")
    for k, v in sample["meta"].items():
        print(f"  {k}: {v}")

    print("[DEBUG] Rain100Dataset OK ✅")
