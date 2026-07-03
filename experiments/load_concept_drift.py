"""Load the same-domain concept-drift splits, ready for training/evaluation.

Run this file and paste your detector training and evaluation code into the marked
region of ``main``: by the time you reach it, the three dataframes you need are
already built for the chosen domain.

For one domain, its query templates are partitioned 50/50 per statement type into
an origin set (S1) and a held-out shifted set (S2). The split yields:

  * ``origin_train``  : benign rows from S1 templates    -> train the detector
  * ``origin_test``   : rows (normal+attack) from S1      -> reference AUROC (eval on S1)
  * ``shifted_test``  : rows from the held-out S2 templates -> post-drift AUROC (eval on S2)

Concept-drift robustness is the AUROC drop between ``origin_test`` and
``shifted_test``. S2 templates are never seen at training time.

The splits are read from the pre-built per-domain CSVs (see
``experiments/build_concept_drift.py``), one ``<domain>.csv`` per domain, with the
three partitions told apart by the (``split``, ``drift_set``) pair:
``(train, origin)`` -> origin_train, ``(test, origin)`` -> origin_test,
``(test, shifted)`` -> shifted_test.

Usage:
    python -m experiments.load_concept_drift                 # all four domains
    python -m experiments.load_concept_drift --domain a      # one domain
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Pre-built concept-drift CSVs (see experiments/build_concept_drift.py).
DATASETS_DIR = Path("~/datasets/superviz26-cd").expanduser()

# Domain letter -> built CSV file name.
DOMAIN_FILE = {
    "a": "a.csv",
    "b": "b.csv",
    "c": "c.csv",
    "d": "d.csv",
}

# Pinned to avoid a DtypeWarning while reading the CSVs. ``drift_set`` is the
# extra column that disambiguates origin_test from shifted_test (both split=test).
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
    "drift_set": str,
}
COLUMNS = list(DTYPES)


def load_concept_drift(
    domain: str,
    datasets_dir: Path = DATASETS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(origin_train, origin_test, shifted_test)`` for one domain letter."""
    path = datasets_dir / DOMAIN_FILE[domain]
    df = pd.read_csv(path, dtype=DTYPES)

    origin = df[df["drift_set"] == "origin"]
    origin_train = origin[origin["split"] == "train"].reset_index(drop=True)
    origin_test = origin[origin["split"] == "test"].reset_index(drop=True)
    shifted_test = (
        df[(df["drift_set"] == "shifted") & (df["split"] == "test")]
        .reset_index(drop=True)
    )

    logger.info(
        "%s: origin_train=%d origin_test=%d shifted_test=%d",
        domain,
        len(origin_train),
        len(origin_test),
        len(shifted_test),
    )
    return origin_train, origin_test, shifted_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default=None,
        choices=list(DOMAIN_FILE),
        help="Domain letter (a..d). Default: iterate over all four domains.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATASETS_DIR,
        help=f"Directory of the pre-built concept-drift CSVs (default: {DATASETS_DIR}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets_dir = args.datasets_dir.expanduser()
    domains = [args.domain] if args.domain else list(DOMAIN_FILE)

    for domain in domains:
        # Benign-only train pool (S1), and the two evaluation pools (S1 / held-out S2).
        origin_train, origin_test, shifted_test = load_concept_drift(
            domain, datasets_dir=datasets_dir
        )
        train_normal = origin_train[origin_train["label"] == 0]

        # ------------------------------------------------------------------ #
        # >>> Paste your training / evaluation code below. <<<
        #
        # Available dataframes (column "full_query" holds the query text,
        # "label" is 0 for benign / 1 for attack):
        #   train_normal  -- benign S1 samples, train the detector on these
        #   origin_test   -- S1 test set,  -> reference AUROC (no drift)
        #   shifted_test  -- held-out S2,  -> post-drift AUROC
        #
        # Drift robustness = AUROC(origin_test) - AUROC(shifted_test).
        # Repeat over all domains (this loop) and average.
        # ------------------------------------------------------------------ #
        _ = train_normal, origin_test, shifted_test  # remove once code is pasted


if __name__ == "__main__":
    main()
