# datasets/pairs.py
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class PairReport:
    name: str
    input_root: str
    gt_root: str
    input_total: int
    gt_total: int
    paired: int
    missing_gt: int
    missing_input: int
    dup_input_keys: int
    dup_gt_keys: int
    example_missing_gt: List[str]
    example_missing_input: List[str]
    example_dup_input: List[str]
    example_dup_gt: List[str]


def _norm(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def list_images(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    files: List[str] = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def make_key_map(
    root: str,
    key_mode: str = "rel_stem",
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build key -> path mapping.
    Returns:
      - map: key -> first path
      - dups: key -> all paths (only for duplicated keys)

    key_mode:
      - "rel_stem": key = relative path (from root) without extension
      - "stem":     key = filename stem only (may collide across folders)
    """
    root = _norm(root)
    files = list_images(root)

    key_to_path: Dict[str, str] = {}
    dups: Dict[str, List[str]] = {}

    for p in files:
        p2 = _norm(p)
        rel = os.path.relpath(p2, root).replace("\\", "/")
        stem = os.path.splitext(rel)[0]
        if key_mode == "stem":
            key = os.path.splitext(os.path.basename(p2))[0]
        else:
            key = stem

        if key in key_to_path:
            if key not in dups:
                dups[key] = [key_to_path[key]]
            dups[key].append(p2)
        else:
            key_to_path[key] = p2

    return key_to_path, dups


def pair_by_key(
    name: str,
    input_root: str,
    gt_root: str,
    input_key_mode: str = "rel_stem",
    gt_key_mode: str = "rel_stem",
    max_examples: int = 10,
) -> Tuple[List[Tuple[str, str, str]], PairReport]:
    """
    Generic pairing: match input and gt by key.
    Returns:
      - pairs: [(key, input_path, gt_path)]
      - report: PairReport summary
    """
    input_root_n = _norm(input_root)
    gt_root_n = _norm(gt_root)

    inp_map, inp_dups = make_key_map(input_root_n, key_mode=input_key_mode)
    gt_map, gt_dups = make_key_map(gt_root_n, key_mode=gt_key_mode)

    inp_keys = set(inp_map.keys())
    gt_keys = set(gt_map.keys())

    common = sorted(list(inp_keys & gt_keys))
    only_inp = sorted(list(inp_keys - gt_keys))
    only_gt = sorted(list(gt_keys - inp_keys))

    pairs = [(k, inp_map[k], gt_map[k]) for k in common]

    rep = PairReport(
        name=name,
        input_root=input_root_n,
        gt_root=gt_root_n,
        input_total=len(inp_map),
        gt_total=len(gt_map),
        paired=len(pairs),
        missing_gt=len(only_inp),
        missing_input=len(only_gt),
        dup_input_keys=len(inp_dups),
        dup_gt_keys=len(gt_dups),
        example_missing_gt=only_inp[:max_examples],
        example_missing_input=only_gt[:max_examples],
        example_dup_input=list(inp_dups.keys())[:max_examples],
        example_dup_gt=list(gt_dups.keys())[:max_examples],
    )
    return pairs, rep


# ============================================================
# Dataset-specific helpers (your directory conventions)
# ============================================================

def scan_raindrop_pairs(
    dataset_name: str,
    drop_root: str,
    clear_root: str,
    max_examples: int = 10,
) -> Tuple[List[Tuple[str, str, str]], PairReport]:
    """
    DayRainDrop / NightRainDrop structure (as provided):
      Drop/<seq>/<frame>.png
      Clear/<seq>/<frame>.png
    Key should include seq + frame (rel_stem), so use rel_stem for both.
    """
    return pair_by_key(
        name=dataset_name,
        input_root=drop_root,
        gt_root=clear_root,
        input_key_mode="rel_stem",
        gt_key_mode="rel_stem",
        max_examples=max_examples,
    )


def scan_deblur_pairs_from_raindrop(
    dataset_name: str,
    blur_root: str,
    clear_root: str,
    max_examples: int = 10,
) -> Tuple[List[Tuple[str, str, str]], PairReport]:
    """
    Deblur example (as provided):
      Blur/<seq>/<frame>.png
      Clear/<seq>/<frame>.png
    """
    return pair_by_key(
        name=dataset_name,
        input_root=blur_root,
        gt_root=clear_root,
        input_key_mode="rel_stem",
        gt_key_mode="rel_stem",
        max_examples=max_examples,
    )


def scan_csd_pairs(
    dataset_name: str,
    snow_root: str,
    gt_root: str,
    max_examples: int = 10,
) -> Tuple[List[Tuple[str, str, str]], PairReport]:
    """
    CSD structure:
      Train/Snow/*.tif
      Train/Gt/*.tif
    Also works if files are in subfolders.
    Key by rel_stem is safe.
    """
    return pair_by_key(
        name=dataset_name,
        input_root=snow_root,
        gt_root=gt_root,
        input_key_mode="rel_stem",
        gt_key_mode="rel_stem",
        max_examples=max_examples,
    )


def scan_rain100_pairs(
    dataset_name: str,
    rain_root: str,
    norain_root: str,
    max_examples: int = 10,
) -> Tuple[List[Tuple[str, str, str]], PairReport]:
    """
    Rain100H structure:
      train/rain/*.png
      train/norain/*.png
    Usually filenames correspond (e.g., rain-1 <-> norain-1) or identical.
    We'll match by STEM ONLY by default to tolerate different folder names.
    If your filenames differ beyond stem, you'll see missing counts.
    """
    return pair_by_key(
        name=dataset_name,
        input_root=rain_root,
        gt_root=norain_root,
        input_key_mode="stem",
        gt_key_mode="stem",
        max_examples=max_examples,
    )


def scan_reside_pairs(
    dataset_name: str,
    hazy_root: str,
    gt_root: str,
    max_examples: int = 10,
) -> Tuple[List[Tuple[str, str, str]], PairReport]:
    """
    RESIDE-6K structure:
      train/hazy/*.jpg
      train/GT/*.jpg
    Filenames match exactly (e.g., 0001_0.8_0.2.jpg).
    Match by STEM is sufficient.
    """
    return pair_by_key(
        name=dataset_name,
        input_root=hazy_root,
        gt_root=gt_root,
        input_key_mode="stem",
        gt_key_mode="stem",
        max_examples=max_examples,
    )
