# datasets/csd.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import List

from datasets.base_dataset import BaseDataset, DatasetDebugConfig


class CSDDataset(BaseDataset):
    """
    CSD Dataset Loader
    Expected structure (variants supported):

    CSD/
      ├─ Train/ or train/
      │   ├─ Snow/
      │   └─ Gt/ or GT/
      └─ Test/ or test/
          ├─ Snow/
          └─ Gt/ or GT/
    """

    degradations = ["snow", "haze", "blur"]

    def _find_existing(self, base: str, candidates: List[str]) -> str:
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                return p
        raise RuntimeError(
            f"[CSDDataset] None of the candidate folders exist under:\n"
            f"  base={base}\n"
            f"  candidates={candidates}"
        )

    def _load_paths(self):
        csd_root = os.path.join(self.root, "CSD")
        if not os.path.isdir(csd_root):
            raise RuntimeError(f"[CSDDataset] CSD root not found: {csd_root}")

        # split folder candidates
        split_dir = self._find_existing(
            csd_root,
            [self.split, self.split.capitalize(), self.split.upper()]
        )

        # input / gt folder candidates
        inp_dir = self._find_existing(
            split_dir,
            ["Snow", "snow", "SNOW"]
        )
        gt_dir = self._find_existing(
            split_dir,
            ["Gt", "GT", "gt", "GroundTruth", "groundtruth"]
        )

        # pair images by filename
        inps, gts, stats = self.pair_by_filename(
            inp_dir=inp_dir,
            gt_dir=gt_dir,
            recursive=False
        )

        self.input_paths = inps
        self.gt_paths = gts


# ------------------------------
# Debug main
# ------------------------------
if __name__ == "__main__":
    """
    Usage:
      python datasets/csd.py

    Make sure your data is located at:
      E:/ReAct-IR/data/CSD/...
    """

    ROOT = "E:/ReAct-IR/data"

    debug_cfg = DatasetDebugConfig(
        enabled=True,
        verbose=True,
        show_first_k=3,
        strict_pairing=True,
        skip_missing_pairs=True,
    )

    print("\n[DEBUG] Initializing CSDDataset...")
    ds = CSDDataset(
        root=ROOT,
        split="train",
        transform=None,
        debug=debug_cfg,
    )

    print(f"\n[DEBUG] Dataset length: {len(ds)}")

    sample = ds[0]
    print("\n[DEBUG] Sample meta:")
    for k, v in sample["meta"].items():
        print(f"  {k}: {v}")

    print("[DEBUG] CSDDataset OK ✅")
