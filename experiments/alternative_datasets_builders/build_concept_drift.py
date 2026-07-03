"""Build the same-domain concept-drift CSVs once, for Zenodo upload.

The concept-drift experiment (``experiments/load_concept_drift.py``) reads its
splits from the pre-built per-domain CSVs produced here. This script derives those
splits from the full per-domain pools (``~/datasets/superviz26-full``), which are too large to
share, and materialises them into one compact CSV per domain so the experiment can
be reproduced from the published data alone.

For each domain it writes ``<domain>.csv`` with three labelled partitions, told
apart by the (``split``, ``drift_set``) pair:

  * origin_train  (split=train, drift_set=origin)   -- benign S1, train the detector
  * origin_test   (split=test,  drift_set=origin)   -- S1 eval, reference AUROC
  * shifted_test  (split=test,  drift_set=shifted)  -- held-out S2, post-drift AUROC

Sizes follow the equal-size scheme of the default in-domain/LODO splits: the
smallest domain (Superviz25-SQL) sets the per-domain budget. Because templates are
split 50/50 into S1 (origin) and S2 (shifted), each side receives half of that
budget, capped by what the domain actually holds:

  * origin_train       = min(335306, N_train) // 2   (~167653 benign rows)
  * origin/shifted_test = min(3353671, N_test) // 2   (~1676835 rows each)

so the two test sides together match the full equal-size test split.

Usage:
    python -m experiments.build_concept_drift                 # all four domains
    python -m experiments.build_concept_drift --domain a      # one domain
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Full per-domain pools (both train and test splits, with template metadata).
FULL_DIR = Path("~/datasets/superviz26-full").expanduser()

# Domain letter -> full-pool filename (file names do not match the letters).
DOMAIN_POOL = {
    "a": "OurAirports.csv",
    "b": "sakila.csv",
    "c": "AdventureWorks.csv",
    "d": "OHR.csv",
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

# The extra column that disambiguates origin_test from shifted_test (both
# split=test). origin_train is part of S1, so it is tagged "origin" too.
DRIFT_COLUMN = "drift_set"
# Inserted just after ``split`` (before the passthrough "ATT&CK label") to match the
# column order of the already-published concept-drift CSVs.
OUT_COLUMNS = COLUMNS[:-1] + [DRIFT_COLUMN, COLUMNS[-1]]

# Equal-size pools: the smallest domain (Superviz25-SQL) sets the per-domain budget,
# so every concept-drift scenario stays the same size as the default in-domain/LODO
# splits. These are the smallest domain's full-pool train/test totals (the global
# minima), used to cap each domain via ``min(EQUAL_*_POOL, N) // 2``.
EQUAL_TRAIN_POOL = 335_306  # smallest domain's full benign train pool
EQUAL_TEST_POOL = 3_353_671  # smallest domain's full test pool
CHUNKSIZE = 500_000


def split_templates_by_statement(
    pool_path: Path, seed: int = 2, chunksize: int = CHUNKSIZE
) -> tuple[set[str], set[str]]:
    """Partition a domain's query templates 50/50 into origin and shifted sets.

    For each ``statement_type`` (SELECT, UPDATE, ...), the distinct
    ``query_template_id`` values seen among the benign train rows are shuffled
    (deterministically, after sorting) and split in half. Splitting per statement
    type keeps both partitions structurally comparable (every statement type is in
    both), so only the concrete templates change between origin and shifted, which
    simulates an abrupt same-domain drift. On an odd count the extra template goes
    to origin. Returns ``(origin_ids, shifted_ids)``.
    """
    pairs: set[tuple[str, str]] = set()
    for chunk in pd.read_csv(pool_path, chunksize=chunksize, dtype=DTYPES):
        mask = (chunk["label"] == 0) & (chunk["split"] == "train")
        sub = chunk[mask][["statement_type", "query_template_id"]].drop_duplicates()
        for row in sub.itertuples(index=False):
            pairs.add((row.statement_type, row.query_template_id))

    by_type: dict[str, list[str]] = {}
    for stype, tid in pairs:
        by_type.setdefault(stype, []).append(tid)

    rng = np.random.default_rng(seed)
    origin_ids: set[str] = set()
    shifted_ids: set[str] = set()
    for tids in by_type.values():
        tids = sorted(tids)  # deterministic order before the shuffle
        rng.shuffle(tids)
        mid = (len(tids) + 1) // 2  # odd count: extra goes to origin
        origin_ids.update(tids[:mid])
        shifted_ids.update(tids[mid:])

    logger.info(
        "%s: %d origin templates, %d shifted templates",
        pool_path.name,
        len(origin_ids),
        len(shifted_ids),
    )
    return origin_ids, shifted_ids


def _sample_by_template_ids(
    pool_path: Path,
    split: str,
    template_ids: set[str],
    n: int,
    seed: int,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    """Sample up to ``n`` rows of ``split`` whose template is in ``template_ids``."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(pool_path, chunksize=chunksize, dtype=DTYPES):
        mask = (chunk["split"] == split) & chunk["query_template_id"].isin(template_ids)
        keep = chunk[mask]
        if not keep.empty:
            parts.append(keep)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    if len(df) > n:
        df = df.sample(n=n, random_state=int(rng.integers(2**31)))
    return df.reset_index(drop=True)


def _count_pool_splits(pool_path: Path, chunksize: int = CHUNKSIZE) -> tuple[int, int]:
    """Return ``(n_benign_train, n_test)`` for a full pool, via one chunked pass.

    Train is benign-only by construction, but we filter on ``label == 0`` explicitly so
    the count matches what ``origin_train`` can draw from. Test counts every row (the
    test split carries both normal and attack samples).
    """
    n_train = n_test = 0
    for chunk in pd.read_csv(pool_path, chunksize=chunksize, usecols=["split", "label"], dtype=DTYPES):
        n_train += int(((chunk["split"] == "train") & (chunk["label"] == 0)).sum())
        n_test += int((chunk["split"] == "test").sum())
    return n_train, n_test


def _derive_domain(
    domain: str,
    train_size: int,
    test_size: int,
    seed: int,
    full_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive ``(origin_train, origin_test, shifted_test)`` from the full pool."""
    pool_path = full_dir / DOMAIN_POOL[domain]
    origin_ids, shifted_ids = split_templates_by_statement(pool_path, seed=seed)

    origin_train = _sample_by_template_ids(pool_path, "train", origin_ids, train_size, seed + 2)
    origin_train["split"] = "train"
    origin_test = _sample_by_template_ids(pool_path, "test", origin_ids, test_size, seed + 3)
    origin_test["split"] = "test"
    shifted_test = _sample_by_template_ids(pool_path, "test", shifted_ids, test_size, seed + 4)
    shifted_test["split"] = "test"

    logger.info(
        "%s: origin_train=%d origin_test=%d shifted_test=%d",
        domain,
        len(origin_train),
        len(origin_test),
        len(shifted_test),
    )
    return origin_train, origin_test, shifted_test


def _build_domain(domain: str, full_dir: Path, seed: int) -> pd.DataFrame:
    """Assemble the materialised concept-drift CSV for one domain letter."""
    pool_path = full_dir / DOMAIN_POOL[domain]
    n_train, n_test = _count_pool_splits(pool_path)

    # Templates are split 50/50 into S1 (origin) and S2 (shifted), so each side gets
    # half of the equal-size budget, capped by what the domain actually holds.
    train_size = min(EQUAL_TRAIN_POOL, n_train) // 2
    test_size = min(EQUAL_TEST_POOL, n_test) // 2
    logger.info(
        "%s: pool benign_train=%d test=%d -> S1 train=%d, per-side test=%d",
        domain,
        n_train,
        n_test,
        train_size,
        test_size,
    )

    origin_train, origin_test, shifted_test = _derive_domain(
        domain, train_size, test_size, seed, full_dir
    )

    origin_train = origin_train.assign(**{DRIFT_COLUMN: "origin"})
    origin_test = origin_test.assign(**{DRIFT_COLUMN: "origin"})
    shifted_test = shifted_test.assign(**{DRIFT_COLUMN: "shifted"})

    # Fail-fast invariant: the held-out S2 templates must never appear in the
    # origin (training/reference) partitions -- that disjointness is the whole
    # point of the drift split.
    origin_tids = set(origin_train["query_template_id"]) | set(origin_test["query_template_id"])
    shifted_tids = set(shifted_test["query_template_id"])
    assert origin_tids.isdisjoint(shifted_tids), f"{domain}: origin/shifted templates overlap"
    assert len(origin_train) and len(origin_test) and len(shifted_test), f"{domain}: empty partition"

    out = pd.concat([origin_train, origin_test, shifted_test], ignore_index=True)
    return out[OUT_COLUMNS]


def build(
    domain: str = "all",
    seed: int = 2,
    full_dir: Path = FULL_DIR,
    out_root: Path = Path("~/datasets/superviz26-cd"),
    overwrite: bool = False,
) -> None:
    """Build the per-domain concept-drift CSVs from the full pools."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    full_dir = full_dir.expanduser()
    out_root = out_root.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    domains = list(DOMAIN_POOL) if domain == "all" else [domain]

    logger.info(f"Building concept-drift sets (seed={seed}) -> {out_root}")
    for name in domains:
        out_path = out_root / f"{name}.csv"
        if out_path.exists() and not overwrite:
            logger.info(f"{name}: {out_path} exists, skipping (use --overwrite).")
            continue
        out = _build_domain(name, full_dir, seed)
        out.to_csv(out_path, index=False)
        logger.info(f"{name}: wrote {len(out)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", default="all", help="Domain letter (a..d) or 'all'."
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Template-partition / sampling seed (match the loader's build)."
    )
    parser.add_argument(
        "--full-dir", type=Path, default=FULL_DIR, help="Directory of the full per-domain pools."
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("~/datasets/superviz26-cd"), help="Output directory for the built CSVs."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild domains whose output CSV already exists."
    )
    args = parser.parse_args()
    build(
        domain=args.domain,
        seed=args.seed,
        full_dir=args.full_dir,
        out_root=args.out_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
