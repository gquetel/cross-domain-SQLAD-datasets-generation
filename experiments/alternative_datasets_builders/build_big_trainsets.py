"""Build the "Big" Superviz26 train sets for the ANUBIS size-sufficiency test.

Each Big scenario doubles the train set of its Superviz26-lodo source: the original
Normal train rows are kept verbatim and an equal number of *novel* benign Extra rows
is appended. The Extra count is read from the source file at build time, so there is
no fixed train size.

This is a single-invocation pipeline: if the novel benign Extra pools are missing, they
are minted automatically (Stage 1, via ``launcher.py --normal-only``) before the Big
CSVs are assembled (Stage 2). Existing pools are reused unless ``--remint`` is given.

Usage:
    python experiments/alternative_datasets_builders/build_big_trainsets.py
    python experiments/alternative_datasets_builders/build_big_trainsets.py --scenario d-d
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Repository root (…/legacy-sqlia-dataset-generator), used to invoke launcher.py for the
# Stage-1 pool minting regardless of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The 11 columns of the Superviz26-SQL CSVs, in canonical order. This repository
# is not installed as a package, so the registry that mlops-sqldetect exposes via
# ``superviz26.manifest()`` is inlined here as the single source of truth.
COLUMNS = [
    "full_query",
    "label",
    "user_inputs",
    "attack_stage",
    "tamper_method",
    "attack_status",
    "statement_type",
    "query_template_id",
    "attack_id",
    "attack_technique",
    "split",
]

# In-domain scenarios (train and test from the same domain) and LODO scenarios
# (train on three domains, test on the held-out one). Inlined replacement for
# ``superviz26.IN_DOMAIN`` / ``superviz26.LODO``.
IN_DOMAIN = ["a-a", "b-b", "c-c", "d-d"]
LODO = ["bcd-a", "acd-b", "abd-c", "abc-d"]

# Scenario name -> the source domains whose benign pools feed its Extra samples.
# Inlined replacement for the manifest's per-file ``train_domains``.
TRAIN_DOMAINS = {
    "a-a": ["a"],
    "b-b": ["b"],
    "c-c": ["c"],
    "d-d": ["d"],
    "bcd-a": ["b", "c", "d"],
    "acd-b": ["a", "c", "d"],
    "abd-c": ["a", "b", "d"],
    "abc-d": ["a", "b", "c"],
}

# Domain letter → full-pool filename under ``--full-root`` (manifest "domains" names
# don't match the file names, so the map is explicit).
DOMAIN_POOL = {
    "a": "OurAirports.csv",
    "b": "sakila.csv",
    "c": "AdventureWorks.csv",
    "d": "OHR.csv",
}


def _per_source_counts(n: int, n_sources: int) -> list[int]:
    """Split ``n`` Extra samples into per-source thirds (remainder to the first sources)."""
    base, rem = divmod(n, n_sources)
    return [base + (1 if i < rem else 0) for i in range(n_sources)]


def _load_pool_extra(pool_path: Path, n: int, seed: int, chunksize: int) -> pd.DataFrame:
    """Sample ``n`` random benign rows from a novel benign-only pool.

    The pool is minted by ``launcher.py --normal-only``: every row is benign
    (``label == 0``, ``split == "train"``) and carries fresh random fills, so the draw
    is genuinely novel relative to the source Normal train. Sampled without replacement
    to keep the Extra rows distinct.
    """
    candidates: list[pd.DataFrame] = []
    for chunk in pd.read_csv(pool_path, chunksize=chunksize, low_memory=False):
        keep = chunk[(chunk["split"] == "train") & (chunk["label"] == 0)]
        if not keep.empty:
            candidates.append(keep)
    pool = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if len(pool) < n:
        raise RuntimeError(f"{pool_path.name}: only {len(pool)} benign rows available (need {n}).")
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def _build_extra(
    sources: list[str], extra_total: int, extra_root: Path, seed: int, chunksize: int
) -> pd.DataFrame:
    """Draw ``extra_total`` novel benign rows for one scenario, split evenly per source domain."""
    counts = _per_source_counts(extra_total, len(sources))
    parts: list[pd.DataFrame] = []
    for i, (domain, n) in enumerate(zip(sources, counts)):
        pool_path = extra_root / DOMAIN_POOL[domain]
        part = _load_pool_extra(pool_path, n, seed=seed + i, chunksize=chunksize)
        logger.info(f"  extra[{domain}]: {len(part)} from {pool_path.name}")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _required_pool_counts(scenarios: list[str], source_root: Path) -> dict[str, int]:
    """Largest benign draw each domain pool must satisfy across ``scenarios``.

    Each scenario re-samples its pools independently, so a pool only needs to be as large
    as the single biggest per-scenario draw against it (not the sum). The in-domain
    scenarios, which pull a whole domain's train count from one pool, dominate this max.
    """
    need: dict[str, int] = {}
    for name in scenarios:
        src = pd.read_csv(source_root / f"{name}.csv", low_memory=False)
        extra_total = int((src["split"] == "train").sum())
        sources = TRAIN_DOMAINS[name]
        for domain, count in zip(sources, _per_source_counts(extra_total, len(sources))):
            need[domain] = max(need.get(domain, 0), count)
    return need


def _mint_pools(extra_root: Path, normal_count: int, config_file: str, no_syn_check: bool) -> None:
    """Mint the novel benign-only per-domain pools by running ``launcher.py --normal-only``.

    The launcher writes one pool CSV per dataset in ``config_file`` (named after the
    dataset, matching ``DOMAIN_POOL``) under ``extra_root`` and spins up an isolated MySQL
    instance per dataset itself, so this builder needs no DB orchestration of its own.
    """
    cmd = [
        sys.executable, "launcher.py",
        "--config-file", config_file,
        "--normal-only", "--normal-count", str(normal_count),
        "--output-dir", str(extra_root),
    ]
    if no_syn_check:
        cmd.append("--no-syn-check")
    logger.info(f"Minting benign pools (normal_count={normal_count}) → {extra_root}")
    logger.info(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _ensure_pools(
    scenarios: list[str], extra_root: Path, source_root: Path, normal_count: int | None,
    config_file: str, no_syn_check: bool, remint: bool,
) -> None:
    """Stage 1: guarantee the benign Extra pools exist (and are large enough) before build.

    Reuses existing pools unless ``remint`` is set. When minting is needed, the pool size
    is auto-derived from the source train counts so the caller never has to guess it; an
    explicit ``normal_count`` overrides this but must cover the largest required draw.
    """
    required = {DOMAIN_POOL[d] for name in scenarios for d in TRAIN_DOMAINS[name]}
    missing = sorted(fn for fn in required if not (extra_root / fn).exists())

    if not missing and not remint:
        logger.info(f"Reusing existing benign pools under {extra_root} (use --remint to rebuild).")
        return

    auto_count = max(_required_pool_counts(scenarios, source_root).values())
    if normal_count is None:
        # Small margin so without-replacement draws never come up one row short.
        count = int(auto_count * 1.02) + 100
    elif normal_count < auto_count:
        raise ValueError(
            f"--normal-count {normal_count} is below the largest required draw ({auto_count}); "
            "raise it or omit it to auto-size."
        )
    else:
        count = normal_count

    if remint and not missing:
        logger.info("Re-minting benign pools (--remint).")
    else:
        logger.info(f"Benign pools missing ({', '.join(missing)}); minting them.")
    extra_root.mkdir(parents=True, exist_ok=True)
    _mint_pools(extra_root, count, config_file, no_syn_check)


def _build_scenario(
    scenario: str, sources: list[str], columns: list[str], source_root: Path, extra_root: Path, seed: int, chunksize: int
) -> pd.DataFrame:
    """Assemble the full Big CSV (doubled train + test) for one scenario.

    The Extra count equals the source Normal train count, so the Big train is exactly
    twice the source train, whatever the source size happens to be.
    """
    src = pd.read_csv(source_root / f"{scenario}.csv", low_memory=False)
    normal = src[src["split"] == "train"].reset_index(drop=True)
    test = src[src["split"] == "test"].reset_index(drop=True)
    extra_total = len(normal)
    logger.info(f"{scenario}: normal={len(normal)} test={len(test)} extra={extra_total} sources={sources}")

    extra = _build_extra(sources, extra_total, extra_root, seed, chunksize)
    big_train = pd.concat([normal, extra], ignore_index=True)
    big_train["split"] = "train"

    # Fail-fast invariants: doubled size, Normal kept verbatim (superset), Extra benign.
    assert len(big_train) == 2 * extra_total, f"{scenario}: big train is {len(big_train)}, expected {2 * extra_total}"
    assert len(extra) == extra_total, f"{scenario}: extra is {len(extra)}, expected {extra_total}"
    # Compare as strings: all-NaN attack columns infer different dtypes per source but
    # write identically to CSV, so value (not dtype) equality is the right invariant.
    front = big_train.iloc[: len(normal)][columns].reset_index(drop=True).astype(str)
    assert front.equals(normal[columns].reset_index(drop=True).astype(str)), f"{scenario}: Normal not preserved in Big"
    assert (extra["label"] == 0).all(), f"{scenario}: Extra contains non-benign rows"

    out = pd.concat([big_train, test], ignore_index=True)
    return out[columns]


def build(
    scenario: str = "all",
    seed: int = 7,
    extra_root: Path = Path("/tmp/superviz26-extra"),
    out_root: Path = Path("~/datasets/superviz26-big"),
    source_root: Path | None = None,
    overwrite: bool = False,
    chunksize: int = 500_000,
    normal_count: int | None = None,
    config_file: str = "config.toml",
    no_syn_check: bool = False,
    remint: bool = False,
) -> None:
    """Build the doubled-train Big CSVs for the size-sufficiency experiment.

    Single invocation: mints the novel benign Extra pools (Stage 1) if they are missing,
    then assembles the Big CSVs (Stage 2). Existing pools are reused unless ``remint``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    extra_root = extra_root.expanduser()
    out_root = out_root.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    src_root = source_root.expanduser() if source_root else (Path.home() / "datasets" / "superviz26-lodo")

    scenarios = IN_DOMAIN + LODO if scenario == "all" else [scenario]

    # Stage 1: mint the throwaway benign pools here rather than making the caller run
    # launcher.py by hand. One launcher run covers every domain referenced by `scenarios`.
    _ensure_pools(scenarios, extra_root, src_root, normal_count, config_file, no_syn_check, remint)

    logger.info(f"Building Big train sets (seed={seed}) → {out_root}")
    for name in scenarios:
        out_path = out_root / f"{name}.csv"
        if out_path.exists() and not overwrite:
            logger.info(f"{name}: {out_path} exists, skipping (use --overwrite).")
            continue
        sources = TRAIN_DOMAINS[name]
        out = _build_scenario(name, sources, COLUMNS, src_root, extra_root, seed, chunksize)
        out.to_csv(out_path, index=False)
        logger.info(f"{name}: wrote {len(out)} rows → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", help="Scenario name (a-a … abc-d) or 'all'.")
    parser.add_argument("--seed", type=int, default=7, help="Base random state for sampling Extra (recorded in logs).")
    parser.add_argument(
        "--extra-root", type=Path, default=Path("/tmp/superviz26-extra"),
        help="Directory of the novel benign-only per-domain pools (from launcher.py --normal-only). "
        "A throwaway scratch location such as /tmp is expected; the pool is not a persistent dataset.",
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("~/datasets/superviz26-big"), help="Output directory for the Big CSVs."
    )
    parser.add_argument(
        "--source-root", type=Path, default=None,
        help="Directory of the source Superviz26-lodo CSVs (default: ~/datasets/superviz26-lodo).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild scenarios whose output CSV already exists."
    )
    parser.add_argument(
        "--chunksize", type=int, default=500_000, help="Rows per chunk when streaming the benign pools."
    )
    parser.add_argument(
        "--normal-count", type=int, default=None,
        help="Pool size to mint per domain in Stage 1 (default: auto-sized from the source "
        "train counts). Must cover the largest required draw if set explicitly.",
    )
    parser.add_argument(
        "--config-file", type=str, default="config.toml",
        help="Config passed to launcher.py when minting the benign pools (Stage 1).",
    )
    parser.add_argument(
        "--no-syn-check", action="store_true",
        help="Skip MySQL syntax validation while minting the benign pools (faster Stage 1).",
    )
    parser.add_argument(
        "--remint", action="store_true",
        help="Re-mint the benign pools even if they already exist under --extra-root.",
    )
    args = parser.parse_args()
    build(
        scenario=args.scenario,
        seed=args.seed,
        extra_root=args.extra_root,
        out_root=args.out_root,
        source_root=args.source_root,
        overwrite=args.overwrite,
        chunksize=args.chunksize,
        normal_count=args.normal_count,
        config_file=args.config_file,
        no_syn_check=args.no_syn_check,
        remint=args.remint,
    )


if __name__ == "__main__":
    main()
