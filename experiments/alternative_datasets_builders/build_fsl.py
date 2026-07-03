"""Build the Superviz26-FSL (few-shot-learning) train sets.

For each in-domain scenario (``a-a`` … ``d-d``) this keeps the in-domain test set
verbatim and replaces the train side with a 100k random subsample of that domain's
own benign train split. The result is a small, fixed-size train budget evaluated
against the full in-domain test set -- the "normal" baseline that
``build_big_trainsets`` reads back from ``~/datasets/superviz26-fsl`` and grows.

The per-domain in-domain CSV (``~/datasets/superviz26-lodo/<domain>.csv``) already
carries both the ``train`` and ``test`` splits, so we read from there and only need
to subsample the train rows; the test rows pass through unchanged.

Usage:
    python -m experiments.alternative_datasets_builders.build_fsl                 # all 4 domains
    python -m experiments.alternative_datasets_builders.build_fsl --scenario a-a  # one domain
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Fixed few-shot train budget: 100k benign rows subsampled from the domain's own
# train split. Matches ``build_big_trainsets.NORMAL_TRAIN_SIZE``.
TRAIN_SIZE = 100_000

# In-domain scenarios: train and test drawn from the same domain.
IN_DOMAIN = ["a-a", "b-b", "c-c", "d-d"]


def _build_scenario(scenario: str, source_root: Path, seed: int) -> pd.DataFrame:
    """Subsample the train split to 100k and keep the in-domain test set verbatim."""
    src = pd.read_csv(source_root / f"{scenario}.csv", low_memory=False)
    train = src[src["split"] == "train"].reset_index(drop=True)
    test = src[src["split"] == "test"].reset_index(drop=True)
    logger.info(f"{scenario}: train pool={len(train)} test={len(test)}")

    if len(train) < TRAIN_SIZE:
        raise RuntimeError(f"{scenario}: only {len(train)} train rows available (need {TRAIN_SIZE}).")
    fsl_train = train.sample(n=TRAIN_SIZE, random_state=seed).reset_index(drop=True)

    # Fail-fast invariants: exact train size and an untouched test set.
    assert len(fsl_train) == TRAIN_SIZE, f"{scenario}: train is {len(fsl_train)}, expected {TRAIN_SIZE}"

    out = pd.concat([fsl_train, test], ignore_index=True)
    return out[src.columns]


def build(
    scenario: str = "all",
    seed: int = 2,
    source_root: Path = Path("~/datasets/superviz26-lodo"),
    out_root: Path = Path("~/datasets/superviz26-fsl"),
    overwrite: bool = False,
) -> None:
    """Build the few-shot-learning train CSVs from the in-domain splits."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source_root = source_root.expanduser()
    out_root = out_root.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    scenarios = IN_DOMAIN if scenario == "all" else [scenario]

    logger.info(f"Building FSL train sets (seed={seed}) -> {out_root}")
    for name in scenarios:
        out_path = out_root / f"{name}.csv"
        if out_path.exists() and not overwrite:
            logger.info(f"{name}: {out_path} exists, skipping (use --overwrite).")
            continue
        out = _build_scenario(name, source_root, seed)
        out.to_csv(out_path, index=False)
        logger.info(f"{name}: wrote {len(out)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", help="In-domain scenario (a-a … d-d) or 'all'.")
    parser.add_argument("--seed", type=int, default=2, help="Random state for subsampling the train split.")
    parser.add_argument(
        "--source-root", type=Path, default=Path("~/datasets/superviz26-lodo"), help="Directory of the in-domain CSVs."
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("~/datasets/superviz26-fsl"), help="Output directory for the FSL CSVs."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild scenarios whose output CSV already exists."
    )
    args = parser.parse_args()
    build(
        scenario=args.scenario,
        seed=args.seed,
        source_root=args.source_root,
        out_root=args.out_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
