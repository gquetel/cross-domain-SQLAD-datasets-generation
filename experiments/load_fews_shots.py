"""Load the few-shot adaptation data, ready for fine-tuning/evaluation.

Run this file and paste your fine-tuning and evaluation code into the marked
region of ``main``: by the time you reach it, the adaptation pool, the
down-sampled test set, and the per-(k, seed) adaptation sample ``df_k`` are
already built for the chosen target domain.

Few-shot adaptation starts from a LODO model (trained on the three source domains)
and adapts it on ``k`` unlabelled benign samples from the *target* domain, then
evaluates on the held-out target test set. The adaptation samples and the test set
both come from the target domain, so they are read from the target's *in-domain*
file (``a-a.csv`` ... ``d-d.csv``). 

This file provides for each k in K_VALUES: 
  * ``df_k``            : the k benign adaptation samples for one (k, seed) run
  * ``df_test``         : target test split, down-sampled to 50k -> evaluate here

Sweep over k in {0, 5, 10, 50, 100, 500, 1000, 10000}; k=0 is the unadapted LODO
model. Recovery is defined as coming within 0.01 AUROC of the in-domain score.

Usage:
    python -m experiments.load_fews_shots                # all four target domains
    python -m experiments.load_fews_shots --target a     # one target domain
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Built scenario CSVs. The target-domain normals + test set live in the in-domain
# file (x-x.csv); see the module docstring for why the LODO file is not used here.
DATASETS_DIR = Path("~/datasets/superviz26-fsl").expanduser()

# Target domain letter -> in-domain scenario file name.
TARGET_FILE = {
    "a": "a-a.csv",
    "b": "b-b.csv",
    "c": "c-c.csv",
    "d": "d-d.csv",
}

# Pinned to avoid a DtypeWarning while streaming the CSVs.
DTYPES = {
    "full_query": str,
    "label": int,
    "user_inputs": str,
    "attack_stage": str,
    "tamper_method": str,
    "attack_status": str,
    "statement_type": str,
    "query_template_id": str,
    "attack_id": str,
    "attack_technique": str,
    "split": str,
    "ATT&CK label": str,
}
COLUMNS = list(DTYPES)

K_VALUES = [0, 5, 10, 50, 100, 500, 1_000, 10_000]
N_RUNS = 5
TEST_SIZE = 50_000
SEED = 2
CHUNKSIZE = 500_000


def load_target_data(
    target: str,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
    datasets_dir: Path = DATASETS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(df_train_normal, df_test)`` for one target domain letter.

    ``df_train_normal`` are the benign rows of the target's in-domain train split
    (the pool the k adaptation samples are drawn from); ``df_test`` is its test
    split (the held-out target domain), down-sampled to ``test_size`` with a fixed
    seed to keep the AUROC computation tractable and reproducible.
    """
    path = datasets_dir / TARGET_FILE[target]
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, dtype=DTYPES):
        tr = chunk[(chunk["split"] == "train") & (chunk["label"] == 0)]
        te = chunk[chunk["split"] == "test"]
        if not tr.empty:
            train_parts.append(tr)
        if not te.empty:
            test_parts.append(te)

    df_train_normal = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pd.DataFrame(columns=COLUMNS)
    )
    df_test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=COLUMNS)
    )
    if len(df_test) > test_size:
        df_test = df_test.sample(n=test_size, random_state=seed).reset_index(drop=True)

    logger.info(
        "%s: train_normal=%d test=%d (attack=%d)",
        target,
        len(df_train_normal),
        len(df_test),
        int((df_test["label"] == 1).sum()),
    )
    return df_train_normal, df_test


def sample_k_normals(df_train_normal: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Draw ``k`` benign adaptation samples for one run (reproducible per seed)."""
    if k > len(df_train_normal):
        raise ValueError(
            f"k={k} exceeds available normal train samples ({len(df_train_normal)})"
        )
    return df_train_normal.sample(n=k, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=None,
        choices=list(TARGET_FILE),
        help="Target domain letter (a..d). Default: iterate over all four targets.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATASETS_DIR,
        help=f"Directory of the built scenario CSVs (default: {DATASETS_DIR}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets_dir = args.datasets_dir.expanduser()
    targets = [args.target] if args.target else list(TARGET_FILE)

    for target in targets:
        # Adaptation pool (benign target train) and the held-out target test set.
        df_train_normal, df_test = load_target_data(target, datasets_dir=datasets_dir)

        for k in K_VALUES:
            # k=0 is the unadapted LODO model evaluated directly on df_test.
            runs = [0] if k == 0 else range(N_RUNS)
            for run in runs:
                seed = SEED + run
                df_k = (
                    df_train_normal.iloc[:0]
                    if k == 0
                    else sample_k_normals(df_train_normal, k, seed)
                )

                # ---------------------------------------------------------- #
                # >>> Paste your fine-tuning / evaluation code below. <<<
                #
                # Available (column "full_query" is the query text, "label" is
                # 0 for benign / 1 for attack):
                #   df_train_normal -- full benign target-train pool
                #   df_k            -- the k benign adaptation samples (empty when k=0)
                #   df_test         -- target test set, evaluate AUROC here
                #
                # k=0: score df_test with the unadapted LODO model.
                # k>0: fine-tune the autoencoder on df_k (extractor frozen),
                #      recompute the threshold from df_k, then score df_test.
                # Average AUROC over the N_RUNS seeds; recovery = within 0.01
                # AUROC of the in-domain score.
                # ---------------------------------------------------------- #
                _ = df_k, df_test  # remove once code is pasted


if __name__ == "__main__":
    main()
