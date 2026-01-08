# datasets/mixed_dataset.py
import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Any

from torch.utils.data import Dataset

# --------------------------------------------------
# Ensure project root is in PYTHONPATH
# --------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.base_dataset import DatasetDebugConfig  # (참고용 / 통일성)
# --------------------------------------------------
# Debug config
# --------------------------------------------------
@dataclass
class MixedDebugConfig:
    enabled: bool = True
    verbose: bool = True
    sample_hist_trials: int = 1000


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _normalize(probs: List[float]) -> List[float]:
    s = sum(probs)
    if s <= 0:
        raise ValueError("[MixedDataset] probs sum must be > 0")
    return [p / s for p in probs]


def _auto_probs(lengths: List[int], mode: str) -> List[float]:
    """
    mode:
      - uniform
      - proportional
      - sqrt  (recommended)
    """
    if mode == "uniform":
        probs = [1.0 for _ in lengths]
    elif mode == "proportional":
        probs = [float(max(1, n)) for n in lengths]
    elif mode == "sqrt":
        probs = [float(max(1, n)) ** 0.5 for n in lengths]
    else:
        raise ValueError(f"[MixedDataset] Unknown balance mode: {mode}")
    return _normalize(probs)


# --------------------------------------------------
# MixedDataset
# --------------------------------------------------
class MixedDataset(Dataset):
    """
    ReAct-IR MixedDataset

    - Mix multiple BaseDataset-compatible datasets
    - Each __getitem__ randomly samples one dataset
    - meta is preserved + annotated with mix_source
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        probs: Optional[Sequence[float]] = None,
        balance: str = "sqrt",
        epoch_length: Optional[int] = None,
        seed: int = 123,
        debug: Optional[Any] = None,  # bool | MixedDebugConfig | None
    ):
        assert len(datasets) > 0, "[MixedDataset] datasets must be non-empty"

        self.datasets = list(datasets)
        self.names = [ds.__class__.__name__ for ds in self.datasets]
        self.lengths = [len(ds) for ds in self.datasets]

        self.seed = seed
        self.rng = random.Random(seed)

        # --------------------------------------------------
        # ✅ Robust debug handling
        # --------------------------------------------------
        if isinstance(debug, bool):
            self.debug = MixedDebugConfig(enabled=debug)
        elif debug is None:
            self.debug = MixedDebugConfig()
        else:
            self.debug = debug  # assume MixedDebugConfig

        # sampling probabilities
        if probs is not None:
            if len(probs) != len(self.datasets):
                raise ValueError("[MixedDataset] probs length mismatch")
            self.probs = _normalize(list(map(float, probs)))
            self.balance = "custom"
        else:
            self.probs = _auto_probs(self.lengths, balance)
            self.balance = balance

        # virtual epoch length
        self.epoch_length = epoch_length if epoch_length is not None else sum(self.lengths)

        if self.debug.enabled:
            self._print_summary()
            self._debug_histogram()

    def __len__(self) -> int:
        return int(self.epoch_length)

    def _choose_dataset_index(self) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(self.probs):
            acc += p
            if r <= acc:
                return i
        return len(self.probs) - 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_i = self._choose_dataset_index()
        ds = self.datasets[ds_i]

        j = self.rng.randrange(len(ds))
        sample = ds[j]

        if not isinstance(sample, dict) or "meta" not in sample:
            raise RuntimeError("[MixedDataset] Dataset must return dict with meta")

        # ensure degradations exists
        sample["meta"].setdefault("degradations", [])

        # annotate
        sample["meta"]["mix_source"] = self.names[ds_i]
        sample["meta"]["mix_prob"] = float(self.probs[ds_i])

        return sample

    # --------------------------------------------------
    # Debug helpers
    # --------------------------------------------------
    def _print_summary(self):
        print("\n[MixedDataset] Summary")
        print(f"  balance     : {self.balance}")
        print(f"  epoch_length: {self.epoch_length}")
        print(f"  seed        : {self.seed}")
        print("  datasets:")
        for i, (n, l, p) in enumerate(zip(self.names, self.lengths, self.probs)):
            print(f"    [{i}] {n:<24} len={l:<7d} prob={p:.4f}")

    def _debug_histogram(self):
        if not self.debug.verbose:
            return
        trials = max(50, int(self.debug.sample_hist_trials))
        counts = [0 for _ in self.datasets]

        tmp_rng = random.Random(self.seed + 999)
        for _ in range(trials):
            r = tmp_rng.random()
            acc = 0.0
            pick = len(self.probs) - 1
            for i, p in enumerate(self.probs):
                acc += p
                if r <= acc:
                    pick = i
                    break
            counts[pick] += 1

        print(f"\n[MixedDataset] Sampling histogram (trials={trials})")
        for i, c in enumerate(counts):
            print(
                f"  {self.names[i]:<24} "
                f"count={c:<5d} ratio={c/trials:.3f} (target={self.probs[i]:.3f})"
            )



# --------------------------------------------------
# Debug main
# --------------------------------------------------
if __name__ == "__main__":
    """
    Usage:
      python datasets/mixed_dataset.py
    """

    from datasets.base_dataset import DatasetDebugConfig
    from datasets.csd import CSDDataset
    from datasets.rain100 import Rain100Dataset
    from datasets.raindrop_day import DayRainDropDataset
    from datasets.raindrop_night import NightRainDropDataset
    from datasets.reside6k import RESIDE6KDataset

    DATA_ROOT = "E:/ReAct-IR/data"

    # silence individual dataset logs
    ds_dbg = DatasetDebugConfig(enabled=False, verbose=False)

    print("\n[DEBUG] Creating datasets...")
    ds_csd = CSDDataset(DATA_ROOT, split="train", debug=ds_dbg)
    ds_r100h = Rain100Dataset(DATA_ROOT, split="train", heavy=True, debug=ds_dbg)
    ds_r100l = Rain100Dataset(DATA_ROOT, split="test", heavy=False, debug=ds_dbg)
    ds_day = DayRainDropDataset(DATA_ROOT, split="train", debug=ds_dbg)
    ds_night = NightRainDropDataset(DATA_ROOT, split="train", debug=ds_dbg)
    ds_reside = RESIDE6KDataset(DATA_ROOT, split="train", debug=ds_dbg)

    mixed = MixedDataset(
        datasets=[ds_csd, ds_r100h, ds_r100l, ds_day, ds_night, ds_reside],
        balance="sqrt",
        epoch_length=20000,
        seed=123,
        debug=MixedDebugConfig(enabled=True, verbose=True, sample_hist_trials=1000),
    )

    print(f"\n[DEBUG] MixedDataset length: {len(mixed)}")

    for i in range(3):
        s = mixed[i]
        print(f"\n[DEBUG] Sample {i}:")
        print(f"  mix_source : {s['meta'].get('mix_source')}")
        print(f"  degradations: {s['meta'].get('degradations')}")
        print(f"  input_path : {s['meta'].get('input_path')}")
        print(f"  gt_path    : {s['meta'].get('gt_path')}")

    print("\n[DEBUG] MixedDataset OK ✅")
