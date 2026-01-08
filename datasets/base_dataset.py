# datasets/base_dataset.py
import os
import sys
import glob
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from torch.utils.data import Dataset
from PIL import Image

# ------------------------------
# Utilities
# ------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS


def safe_open_rgb(path: str) -> Image.Image:
    """
    Open an image path with PIL and convert to RGB.
    Raises with an informative message if it fails.
    """
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"[BaseDataset] Failed to open image: {path}\n  Error: {e}")


def list_images_recursive(folder: str) -> List[str]:
    """
    Recursively list image files under folder.
    """
    if not os.path.isdir(folder):
        return []
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            p = os.path.join(root, n)
            if is_image_file(p):
                files.append(p)
    return sorted(files)


@dataclass
class DatasetDebugConfig:
    enabled: bool = True
    verbose: bool = True
    show_first_k: int = 5
    strict_pairing: bool = True  # if True: lengths must match
    skip_missing_pairs: bool = True  # if True: skip if GT missing
    raise_on_corrupt: bool = False  # if True: throw error when an image can't be opened
    max_scan: Optional[int] = None  # None means scan all; set int for quick debug


# ------------------------------
# BaseDataset
# ------------------------------
class BaseDataset(Dataset):
    """
    Base dataset for ReAct-IR.

    Each sample must return:
      {
        "input": <PIL.Image or Tensor>,
        "gt": <PIL.Image or Tensor>,
        "meta": {
            "dataset": str,
            "degradations": list[str],
            "input_path": str,
            "gt_path": str
        }
      }

    Child class MUST implement:
      - self.degradations (class attribute)
      - _load_paths() to fill self.input_paths and self.gt_paths

    Notes:
      - transform signature: transform(inp, gt) -> (inp, gt)
      - This file includes a runnable debug main.
    """

    degradations: List[str] = []  # must override in child

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        debug: Optional[DatasetDebugConfig] = None,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.debug = debug or DatasetDebugConfig()

        self.input_paths: List[str] = []
        self.gt_paths: List[str] = []

        self._load_paths()

        # Basic sanity checks + debug prints
        self._post_check_and_debug()

    # -------- required override ----------
    def _load_paths(self) -> None:
        raise NotImplementedError

    # -------- framework methods ----------
    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, idx: int) -> Dict:
        inp_path = self.input_paths[idx]
        gt_path = self.gt_paths[idx]

        try:
            inp = safe_open_rgb(inp_path)
            gt = safe_open_rgb(gt_path)
        except Exception as e:
            if self.debug.raise_on_corrupt:
                raise
            # fallback: return a different sample (deterministic-ish)
            # NOTE: for true training you'd want a better strategy, but for debugging this is fine.
            new_idx = 0 if idx != 0 else min(1, len(self.input_paths) - 1)
            if self.debug.enabled and self.debug.verbose:
                print(f"[BaseDataset] WARNING: corrupt sample at idx={idx}. Switching to idx={new_idx}")
                print(str(e))
            inp_path = self.input_paths[new_idx]
            gt_path = self.gt_paths[new_idx]
            inp = safe_open_rgb(inp_path)
            gt = safe_open_rgb(gt_path)

        if self.transform is not None:
            out = self.transform(inp, gt)
            # allow both tuple return or dict return
            if isinstance(out, tuple) and len(out) == 2:
                inp, gt = out
            else:
                raise RuntimeError(
                    "[BaseDataset] transform must return (inp, gt). "
                    f"Got type={type(out)}"
                )

        return {
            "input": inp,
            "gt": gt,
            "meta": {
                "dataset": self.__class__.__name__,
                "degradations": list(self.degradations),
                "input_path": inp_path,
                "gt_path": gt_path,
            },
        }

    # -------- debug helpers ----------
    def _post_check_and_debug(self) -> None:
        if self.debug.enabled:
            print(f"\n[Dataset] {self.__class__.__name__}")
            print(f"  root : {self.root}")
            print(f"  split: {self.split}")
            print(f"  degradations: {self.degradations}")
            print(f"  #pairs: {len(self.input_paths)}")

        if len(self.input_paths) == 0 or len(self.gt_paths) == 0:
            raise RuntimeError(
                f"[BaseDataset] Empty dataset paths.\n"
                f"  class={self.__class__.__name__}\n"
                f"  root={self.root}\n"
                f"  split={self.split}\n"
                f"  input_paths={len(self.input_paths)} gt_paths={len(self.gt_paths)}\n"
                f"  -> Check folder names / split / extensions."
            )

        if self.debug.strict_pairing and len(self.input_paths) != len(self.gt_paths):
            raise RuntimeError(
                f"[BaseDataset] Pair count mismatch.\n"
                f"  input={len(self.input_paths)} gt={len(self.gt_paths)}\n"
                f"  class={self.__class__.__name__}"
            )

        if self.debug.enabled and self.debug.verbose:
            k = min(self.debug.show_first_k, len(self.input_paths))
            print(f"  show first {k} pairs:")
            for i in range(k):
                print(f"    [{i:04d}] IN : {self.input_paths[i]}")
                print(f"           GT : {self.gt_paths[i]}")

    # -------- optional pairing utility (useful for child classes) ----------
    def pair_by_filename(
        self,
        inp_dir: str,
        gt_dir: str,
        recursive: bool = False,
        inp_glob: str = "*",
        gt_glob: str = "*",
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Pair images by filename (basename matching).
        Returns (input_paths, gt_paths, stats)
        """
        if recursive:
            inp_files = list_images_recursive(inp_dir)
            gt_files = list_images_recursive(gt_dir)
        else:
            inp_files = sorted([p for p in glob.glob(os.path.join(inp_dir, inp_glob)) if is_image_file(p)])
            gt_files = sorted([p for p in glob.glob(os.path.join(gt_dir, gt_glob)) if is_image_file(p)])

        if self.debug.max_scan is not None:
            inp_files = inp_files[: self.debug.max_scan]
            gt_files = gt_files[: self.debug.max_scan]

        gt_map = {os.path.basename(p): p for p in gt_files}

        paired_in, paired_gt = [], []
        missing = 0
        for p in inp_files:
            bn = os.path.basename(p)
            if bn in gt_map:
                paired_in.append(p)
                paired_gt.append(gt_map[bn])
            else:
                missing += 1
                if not self.debug.skip_missing_pairs:
                    raise RuntimeError(f"[BaseDataset] Missing GT for input: {p}")

        stats = {
            "inp_total": len(inp_files),
            "gt_total": len(gt_files),
            "paired": len(paired_in),
            "missing_gt_for_inp": missing,
        }

        if self.debug.enabled:
            print(f"  [pair_by_filename] inp_total={stats['inp_total']} gt_total={stats['gt_total']} paired={stats['paired']}")
            if missing > 0:
                print(f"  [pair_by_filename] WARNING: missing_gt_for_inp={missing}")

        return paired_in, paired_gt, stats


# ------------------------------
# Debug main
# ------------------------------
def _debug_print_sample(sample: Dict) -> None:
    inp = sample["input"]
    gt = sample["gt"]
    meta = sample["meta"]

    # handle PIL or Tensor
    def _shape(x):
        if hasattr(x, "size") and isinstance(x, Image.Image):
            return f"PIL{tuple(x.size)}"
        if hasattr(x, "shape"):
            return f"TENSOR{tuple(x.shape)}"
        return str(type(x))

    print("\n[Sample]")
    print(f"  input: {_shape(inp)}")
    print(f"  gt   : {_shape(gt)}")
    print(f"  meta : {meta}")


if __name__ == "__main__":
    print("[base_dataset.py] OK. This file defines BaseDataset with debug utilities.")
    print("[base_dataset.py] Next: implement child dataset files and run them directly.")
