# scripts/inspect_pairs.py
import os
import sys
import argparse
from typing import List

# --------------------------------------------------
# Ensure project root (E:/ReAct-IR) is in sys.path
# so "from datasets..." works when running directly.
# --------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))              # .../scripts
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))       # .../ReAct-IR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.pairs import (
    PairReport,
    scan_raindrop_pairs,
    scan_deblur_pairs_from_raindrop,
    scan_csd_pairs,
    scan_rain100_pairs,
    scan_reside_pairs,
)


def print_report(rep: PairReport, show_examples: bool = True):
    print(f"\n=== [{rep.name}] ===")
    print(f"input_root: {rep.input_root}")
    print(f"gt_root   : {rep.gt_root}")
    print(f"input_total={rep.input_total}  gt_total={rep.gt_total}")
    print(f"paired={rep.paired}  missing_gt={rep.missing_gt}  missing_input={rep.missing_input}")
    print(f"dup_input_keys={rep.dup_input_keys}  dup_gt_keys={rep.dup_gt_keys}")

    if show_examples:
        if rep.missing_gt > 0:
            print("  - example_missing_gt(keys):")
            for k in rep.example_missing_gt:
                print("    ", k)
        if rep.missing_input > 0:
            print("  - example_missing_input(keys):")
            for k in rep.example_missing_input:
                print("    ", k)
        if rep.dup_input_keys > 0:
            print("  - example_dup_input_keys:")
            for k in rep.example_dup_input:
                print("    ", k)
        if rep.dup_gt_keys > 0:
            print("  - example_dup_gt_keys:")
            for k in rep.example_dup_gt:
                print("    ", k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="E:/ReAct-IR/data", help="Root folder containing datasets")
    ap.add_argument("--max_examples", type=int, default=10)
    ap.add_argument("--no_examples", action="store_true", help="Do not print example keys")
    ap.add_argument("--only", default="", help="Comma list: deraindrop_day,deraindrop_night,deblur,csd,rain100h,reside")
    a = ap.parse_args()

    show_examples = not a.no_examples
    only_set = set([s.strip().lower() for s in a.only.split(",") if s.strip()])

    def want(name: str) -> bool:
        return (len(only_set) == 0) or (name.lower() in only_set)

    root = os.path.normpath(a.data_root)

    # 1) deraindrop (day)
    if want("deraindrop_day"):
        drop = os.path.join(root, "DayRainDrop", "Drop")
        clear = os.path.join(root, "DayRainDrop", "Clear")
        _, rep = scan_raindrop_pairs("deraindrop_day", drop, clear, max_examples=a.max_examples)
        print_report(rep, show_examples=show_examples)

    # 2) deraindrop (night)
    if want("deraindrop_night"):
        drop = os.path.join(root, "NightRainDrop", "Drop")
        clear = os.path.join(root, "NightRainDrop", "Clear")
        _, rep = scan_raindrop_pairs("deraindrop_night", drop, clear, max_examples=a.max_examples)
        print_report(rep, show_examples=show_examples)

    # 3) deblur (from DayRainDrop example)
    if want("deblur"):
        blur = os.path.join(root, "DayRainDrop", "Blur")
        clear = os.path.join(root, "DayRainDrop", "Clear")
        _, rep = scan_deblur_pairs_from_raindrop("deblur_dayraindrop", blur, clear, max_examples=a.max_examples)
        print_report(rep, show_examples=show_examples)

    # 4) desnow (CSD) - scan both Train and Test explicitly
    if want("csd"):
        csd_root = os.path.join(root, "CSD")
        # Train
        snow_tr = os.path.join(csd_root, "Train", "Snow")
        gt_tr = os.path.join(csd_root, "Train", "Gt")
        _, rep_tr = scan_csd_pairs("desnow_csd_train", snow_tr, gt_tr, max_examples=a.max_examples)
        print_report(rep_tr, show_examples=show_examples)

        # Test (inspection only; do not use for training)
        snow_te = os.path.join(csd_root, "Test", "Snow")
        gt_te = os.path.join(csd_root, "Test", "Gt")
        _, rep_te = scan_csd_pairs("desnow_csd_test", snow_te, gt_te, max_examples=a.max_examples)
        print_report(rep_te, show_examples=show_examples)

    # 5) derain (Rain100H) - scan both train and test
    if want("rain100h"):
        rroot = os.path.join(root, "rain100H")

        rain_tr = os.path.join(rroot, "train", "rain")
        gt_tr = os.path.join(rroot, "train", "norain")
        _, rep_tr = scan_rain100_pairs("derain_rain100H_train", rain_tr, gt_tr, max_examples=a.max_examples)
        print_report(rep_tr, show_examples=show_examples)

        rain_te = os.path.join(rroot, "test", "rain")
        gt_te = os.path.join(rroot, "test", "norain")
        _, rep_te = scan_rain100_pairs("derain_rain100H_test", rain_te, gt_te, max_examples=a.max_examples)
        print_report(rep_te, show_examples=show_examples)

    # 6) dehaze (RESIDE-6K) - scan both train and test
    if want("reside"):
        rroot = os.path.join(root, "RESIDE-6K")

        hazy_tr = os.path.join(rroot, "train", "hazy")
        gt_tr = os.path.join(rroot, "train", "GT")
        _, rep_tr = scan_reside_pairs("dehaze_reside6k_train", hazy_tr, gt_tr, max_examples=a.max_examples)
        print_report(rep_tr, show_examples=show_examples)

        hazy_te = os.path.join(rroot, "test", "hazy")
        gt_te = os.path.join(rroot, "test", "GT")
        _, rep_te = scan_reside_pairs("dehaze_reside6k_test", hazy_te, gt_te, max_examples=a.max_examples)
        print_report(rep_te, show_examples=show_examples)

    print("\n[inspect_pairs] done.")


if __name__ == "__main__":
    main()
