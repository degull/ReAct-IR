# datasets/raindrop_night.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from typing import Dict, List, Optional, Tuple

from datasets.base_dataset import BaseDataset, DatasetDebugConfig, is_image_file


class NightRainDropDataset(BaseDataset):
    """
    NightRainDrop Dataset Loader (robust, auto)

    Supports BOTH:
    (A) Class-folder layout (like DayRainDrop):
        NightRainDrop/
          ├─ Clear/   (GT)
          ├─ Drop/    (degraded)
          ├─ Blur/    (degraded)
          ├─ Noise/   (optional)
          ├─ Lowlight/(optional)
          └─ ...

        - Pairs are created by matching relative paths to Clear.

    (B) Paired layout:
        NightRainDrop/
          ├─ train/ or Train/
          │   ├─ input/ (or rainy, etc.)
          │   └─ gt/
          └─ test/
              ├─ input/
              └─ gt/
    """

    # nominal set for paper description; per-sample label is dynamic
    degradations = ["drop", "blur", "noise", "lowlight"]

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    def _collect_files(self, root: str) -> Dict[str, str]:
        mapping = {}
        for cur_root, _, files in os.walk(root):
            for f in files:
                if not is_image_file(f):
                    continue
                abs_path = os.path.join(cur_root, f)
                rel_path = os.path.relpath(abs_path, root)
                mapping[rel_path] = abs_path
        return mapping

    def _find_existing_case_insensitive(self, base: str, candidates: List[str]) -> Optional[str]:
        """
        Return first existing directory among candidates (case-insensitive search).
        """
        if not os.path.isdir(base):
            return None

        # direct check first
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                return p

        # case-insensitive check by listing dirs
        subdirs = []
        try:
            subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        except Exception:
            subdirs = []

        low_map = {d.lower(): d for d in subdirs}
        for c in candidates:
            key = c.lower()
            if key in low_map:
                return os.path.join(base, low_map[key])

        return None

    def _resolve_split_root(self, base_root: str) -> str:
        """
        Return split directory if exists; otherwise base_root.
        """
        for c in [self.split, self.split.capitalize(), self.split.upper()]:
            p = os.path.join(base_root, c)
            if os.path.isdir(p):
                return p
        return base_root

    # --------------------------------------------------
    # strategy A: class folders
    # --------------------------------------------------
    def _try_class_folder_layout(self, base_root: str) -> bool:
        """
        Detect and load class folder layout:
          Clear + multiple degradation folders.
        """
        # Clear folder candidates
        clear_dir = self._find_existing_case_insensitive(base_root, ["Clear", "GT", "Gt", "clean", "Clean"])
        if clear_dir is None:
            return False

        # Degradation folder candidates (we accept multiple)
        # You can extend this list later without breaking anything.
        deg_candidates = [
            ("drop", ["Drop", "Raindrop", "RainDrop", "Drops"]),
            ("blur", ["Blur", "Blurry"]),
            ("noise", ["Noise", "Noisy"]),
            ("lowlight", ["Lowlight", "LowLight", "dark", "Dark"]),
            ("rain", ["Rain", "Rainy"]),
            ("haze", ["Haze", "Fog", "fog"]),
            ("jpeg", ["JPEG", "Jpeg", "Artifacts"]),
        ]

        deg_dirs = []
        for label, names in deg_candidates:
            d = self._find_existing_case_insensitive(base_root, names)
            if d is not None and os.path.abspath(d) != os.path.abspath(clear_dir):
                deg_dirs.append((label, d))

        if len(deg_dirs) == 0:
            # Clear exists but no degradation dirs found
            return False

        if self.debug.enabled:
            print("[NightRainDropDataset] Detected class-folder layout.")
            print(f"  Clear(GT): {clear_dir}")
            for lbl, d in deg_dirs:
                print(f"  {lbl.upper():>7}: {d}")

        clear_files = self._collect_files(clear_dir)
        if len(clear_files) == 0:
            raise RuntimeError(f"[NightRainDropDataset] Clear folder has no images: {clear_dir}")

        input_paths, gt_paths = [], []
        sample_degs: List[List[str]] = []

        # For each degradation folder, pair rel paths with Clear
        per_label_counts = {}
        for label, ddir in deg_dirs:
            deg_files = self._collect_files(ddir)
            cnt = 0
            for rel, gt_path in clear_files.items():
                if rel in deg_files:
                    input_paths.append(deg_files[rel])
                    gt_paths.append(gt_path)
                    sample_degs.append([label])
                    cnt += 1
            per_label_counts[label] = cnt

        if len(input_paths) == 0:
            raise RuntimeError(
                "[NightRainDropDataset] Class-layout detected but no pairs found.\n"
                f"  Clear: {clear_dir}\n"
                f"  Degradation dirs: {[d for _, d in deg_dirs]}\n"
                "  -> Check if relative subfolder structure matches between folders."
            )

        self.input_paths = input_paths
        self.gt_paths = gt_paths
        self.sample_degradations = sample_degs

        if self.debug.enabled:
            print(f"[NightRainDropDataset] Total pairs: {len(self.input_paths)}")
            for k, v in per_label_counts.items():
                print(f"  {k:>7} pairs: {v}")

        return True

    # --------------------------------------------------
    # strategy B: paired folders (fallback)
    # --------------------------------------------------
    def _try_paired_layout(self, base_root: str) -> bool:
        """
        Paired layout:
          <base>/<split>/input and <base>/<split>/gt
        """
        split_root = self._resolve_split_root(base_root)

        # input/gt candidates
        inp_dir = self._find_existing_case_insensitive(
            split_root, ["input", "Input", "rain", "Rain", "images", "Images", "degraded", "Degraded"]
        )
        gt_dir = self._find_existing_case_insensitive(
            split_root, ["gt", "GT", "Gt", "clear", "Clear", "clean", "Clean", "target", "Target"]
        )

        if inp_dir is None or gt_dir is None:
            return False

        if self.debug.enabled:
            print("[NightRainDropDataset] Detected paired-folder layout.")
            print(f"  input: {inp_dir}")
            print(f"  gt   : {gt_dir}")

        inps, gts, _ = self.pair_by_filename(inp_dir=inp_dir, gt_dir=gt_dir, recursive=False)
        if len(inps) == 0:
            raise RuntimeError(
                "[NightRainDropDataset] Paired layout found but pairing failed (0 pairs).\n"
                f"  input_dir={inp_dir}\n"
                f"  gt_dir={gt_dir}"
            )

        self.input_paths = inps
        self.gt_paths = gts
        # degradation label unknown in paired layout → mark as 'drop' by default (night raindrop dataset intent)
        self.sample_degradations = [["drop"] for _ in range(len(inps))]
        return True

    # --------------------------------------------------
    # required override
    # --------------------------------------------------
    def _load_paths(self):
        base_root = os.path.join(self.root, "NightRainDrop")
        if not os.path.isdir(base_root):
            raise RuntimeError(f"[NightRainDropDataset] Root not found: {base_root}")

        if self.debug.enabled:
            print(f"[NightRainDropDataset] base={base_root}")

        # Try A then B
        ok = self._try_class_folder_layout(base_root)
        if not ok:
            ok = self._try_paired_layout(base_root)

        if not ok:
            # helpful hint: list top-level dirs
            top_dirs = []
            try:
                top_dirs = [d for d in os.listdir(base_root) if os.path.isdir(os.path.join(base_root, d))]
            except Exception:
                pass

            raise RuntimeError(
                "[NightRainDropDataset] Could not detect dataset layout.\n"
                f"  base_root={base_root}\n"
                f"  top_level_dirs={top_dirs}\n"
                "  Expected either:\n"
                "   (A) Clear/ + (Drop/Blur/Noise/Lowlight...)\n"
                "   (B) split/input + split/gt\n"
            )

    # override to inject per-sample degradation label
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # ensure exists
        if hasattr(self, "sample_degradations"):
            sample["meta"]["degradations"] = self.sample_degradations[idx]
        return sample


# --------------------------------------------------
# Debug main
# --------------------------------------------------
if __name__ == "__main__":
    """
    Usage:
      python datasets/raindrop_night.py

    Expected:
      E:/ReAct-IR/data/NightRainDrop/...
    """
    ROOT = "E:/ReAct-IR/data"

    debug_cfg = DatasetDebugConfig(
        enabled=True,
        verbose=True,
        show_first_k=5,
        strict_pairing=True,
        skip_missing_pairs=True,
    )

    print("\n[DEBUG] Initializing NightRainDropDataset...")
    ds = NightRainDropDataset(
        root=ROOT,
        split="train",
        transform=None,
        debug=debug_cfg
    )

    print(f"\n[DEBUG] Dataset length: {len(ds)}")

    for i in range(min(3, len(ds))):
        s = ds[i]
        print(f"\n[DEBUG] Sample {i}:")
        for k, v in s["meta"].items():
            print(f"  {k}: {v}")

    print("[DEBUG] NightRainDropDataset OK ✅")
