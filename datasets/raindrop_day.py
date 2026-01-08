# datasets/raindrop_day.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from typing import Dict, List, Tuple

from datasets.base_dataset import BaseDataset, DatasetDebugConfig, is_image_file


class DayRainDropDataset(BaseDataset):
    """
    DayRainDrop Dataset (class-based degradation folders)

    Structure:
      DayRainDrop/
        ├─ Clear/   ← GT
        ├─ Drop/    ← raindrop degraded
        └─ Blur/    ← blur degraded

    Each degraded folder mirrors Clear's subfolder structure.
    """

    # NOTE: actual degradation for each sample is dynamic
    degradations = ["drop", "blur"]

    def _collect_files(self, root: str) -> Dict[str, str]:
        """
        Collect all image files under root.
        Returns mapping: relative_path -> absolute_path
        """
        mapping = {}
        for cur_root, _, files in os.walk(root):
            for f in files:
                if not is_image_file(f):
                    continue
                abs_path = os.path.join(cur_root, f)
                rel_path = os.path.relpath(abs_path, root)
                mapping[rel_path] = abs_path
        return mapping

    def _load_paths(self):
        base = os.path.join(self.root, "DayRainDrop")
        if not os.path.isdir(base):
            raise RuntimeError(f"[DayRainDropDataset] Root not found: {base}")

        clear_dir = os.path.join(base, "Clear")
        drop_dir  = os.path.join(base, "Drop")
        blur_dir  = os.path.join(base, "Blur")

        for d in [clear_dir, drop_dir, blur_dir]:
            if not os.path.isdir(d):
                raise RuntimeError(f"[DayRainDropDataset] Missing folder: {d}")

        if self.debug.enabled:
            print("[DayRainDropDataset] Using structure:")
            print(f"  Clear: {clear_dir}")
            print(f"  Drop : {drop_dir}")
            print(f"  Blur : {blur_dir}")

        # Collect files
        clear_files = self._collect_files(clear_dir)
        drop_files  = self._collect_files(drop_dir)
        blur_files  = self._collect_files(blur_dir)

        input_paths = []
        gt_paths = []
        self.sample_degradations = []  # per-sample degradation label

        # Pair Drop -> Clear
        for rel, gt_path in clear_files.items():
            if rel in drop_files:
                input_paths.append(drop_files[rel])
                gt_paths.append(gt_path)
                self.sample_degradations.append(["drop"])

        # Pair Blur -> Clear
        for rel, gt_path in clear_files.items():
            if rel in blur_files:
                input_paths.append(blur_files[rel])
                gt_paths.append(gt_path)
                self.sample_degradations.append(["blur"])

        if len(input_paths) == 0:
            raise RuntimeError(
                "[DayRainDropDataset] No paired samples found.\n"
                "Expected matching relative paths between Clear and Drop/Blur."
            )

        self.input_paths = input_paths
        self.gt_paths = gt_paths

        if self.debug.enabled:
            print(f"[DayRainDropDataset] Total pairs: {len(self.input_paths)}")
            print(f"  Drop pairs: {sum('drop' in d for d in self.sample_degradations)}")
            print(f"  Blur pairs: {sum('blur' in d for d in self.sample_degradations)}")

    # override to inject per-sample degradation
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["meta"]["degradations"] = self.sample_degradations[idx]
        return sample


# --------------------------------------------------
# Debug main
# --------------------------------------------------
if __name__ == "__main__":
    """
    Usage:
      python datasets/raindrop_day.py
    """

    ROOT = "E:/ReAct-IR/data"

    debug_cfg = DatasetDebugConfig(
        enabled=True,
        verbose=True,
        show_first_k=5,
        strict_pairing=True,
        skip_missing_pairs=True,
    )

    print("\n[DEBUG] Initializing DayRainDropDataset (FINAL)...")
    ds = DayRainDropDataset(
        root=ROOT,
        split="train",   # split not used (class-based dataset)
        transform=None,
        debug=debug_cfg
    )

    print(f"\n[DEBUG] Dataset length: {len(ds)}")

    for i in range(3):
        sample = ds[i]
        print(f"\n[DEBUG] Sample {i}:")
        for k, v in sample["meta"].items():
            print(f"  {k}: {v}")

    print("[DEBUG] DayRainDropDataset OK ✅")
