import argparse
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sqlglot
import sqlglot.errors
import sqlparse
import sys
import torch
from scipy.stats import gmean
from scipy.spatial.distance import pdist

from transformers import RobertaTokenizerFast, RobertaModel
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer


# Resolve the dataset directory relative to the current user's home. 
# Change to actual location of datasets.
DATASETS_DIR = Path.home() / "datasets" / "superviz26-fsl"

# dtype is specified to prevent a DtypeWarning when reading the CSVs.
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
}


def print_vocab_size(queries, type: str, name: str) -> dict:
    """Lexical diversity: vocabulary size and Type-Token Ratio.

    Returns a dict of the metrics so the caller can aggregate them into the
    per-mode results CSV (the per-vocab .txt dump is kept as a side artifact).
    """
    if not queries:
        print(f"Vocabulary for {name} {type} queries: skipped (empty pool).")
        return {"vocab_size": 0, "token_count": 0, "ttr": float("nan")}

    v = CountVectorizer()
    X = v.fit_transform(queries)
    vocab_size = len(v.vocabulary_)
    print(f"Vocabulary size for {name} {type} queries: {vocab_size}")

    token_count = int(X.sum())
    ttr = vocab_size / token_count if token_count else 0
    print(f"Type-Token Ratio (TTR) for {name} {type} queries: {ttr:.4f}")

    with open(f"vocab-{name}-{type}.txt", "w") as f:
        for word, idx in sorted(v.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {word}\n")

    return {"vocab_size": vocab_size, "token_count": token_count, "ttr": ttr}


def print_unique_pts(queries: list, type: str, name: str) -> dict:
    pts = {}
    cnt_prserr = 0

    logging.disable(sys.maxsize)
    for q in tqdm(queries):
        try:
            glot_trees = sqlglot.parse(q, dialect="mysql")
            for glot_tree in glot_trees:
                if glot_tree == None or isinstance(glot_tree, sqlglot.exp.Command):
                    # A Command is returned, the tool didn't manage to parse the query
                    # correctly, ignore those.
                    cnt_prserr += 1
                    continue

                # Replace all literals or identifier to get a canonical representation.
                # "Normalize" parse trees.
                for i in glot_tree.find_all(
                    sqlglot.exp.Identifier | sqlglot.exp.Literal | sqlglot.exp.Comment
                ):
                    i.set("this", "I")

                for i in glot_tree.find_all(sqlglot.exp.HexString):
                    i.set("this", "0")

                # print(repr(glot_tree))
                canon_tree = glot_tree.sql(comments=False)
                if canon_tree not in pts:
                    pts[canon_tree] = 1
                else:
                    pts[canon_tree] += 1
        except sqlglot.errors.ParseError as e:
            cnt_prserr += 1
        except sqlglot.errors.TokenError as e:
            cnt_prserr += 1
        except KeyError as e:
            cnt_prserr += 1

    logging.disable(logging.NOTSET)

    if cnt_prserr > 0:
        print(f"There were {cnt_prserr} parsing errors during processing.")
    s_keys = sorted(pts)
    with open(f"parse-trees-{name}-{type}.txt", "w") as f:
        for e in s_keys:
            f.write(f"{e}: {pts[e]}\n")
    print(f"Number of unique parse trees for {name} {type} queries: {len(pts)}")

    return {"n_unique_parse_trees": len(pts), "n_parse_errors": cnt_prserr}


def compute_embeddings(df: pd.DataFrame):
    """Compute SecureBERT embeddings of queries (column 'full_query').

    This is a one-time experiment, so embeddings are recomputed every call (no
    caching).

    Args:
        df (pd.DataFrame): frame with a 'full_query' column to embed.
    """
    queries = df["full_query"].to_list()

    bert_model = "ehsanaghaei/SecureBERT"
    tokenizer = RobertaTokenizerFast.from_pretrained(bert_model)
    rb_model = RobertaModel.from_pretrained(bert_model)
    rb_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rb_model.to(device)
    # We compute embeddings by batches, they should not be too big because
    # they might be bigger than memory.
    embeddings = []

    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i : i + batch_size]

            inputs = tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move inputs to device and get embeddings.
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Move back to CPU and convert to numpy
            outputs = rb_model(**inputs, output_hidden_states=True)
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def print_dataset_tsne(
    df: pd.DataFrame, type: str, name: str, n_sampling: None | int = None
):
    if n_sampling:
        df = df.sample(n_sampling, random_state=42)

    queries = df["full_query"].to_list()
    embeddings = compute_embeddings(df)

    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # Let's use default params as much as possible.
    # We set perplexity to 50 as the doc states that higher dimensions requires
    # higher values.
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(50, len(queries) - 1),
        verbose=1,
        n_jobs=-1,
    )
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Save the results to allow to compute the figure with all datasets later.
    results = {
        "queries": queries,
        "embeddings": embeddings,
        "tsne_embeddings": tsne_embeddings,
        "type": type,
        "name": name,
    }

    print(f"t-SNE results saved to tsne-{name}-{type}.pkl")

    with open(f"../output/tsne-{name}-{type}.pkl", "wb") as f:
        pickle.dump(results, f)

    # Now plot individual results.
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        alpha=0.6,
        s=20,
    )

    plt.title(
        f"t-SNE Visualization of {name} {type} \n"
        f"Using SecureBERT Embeddings (n={len(queries)})"
    )
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    legend_label = f"{type.capitalize()} Queries"
    plt.legend([scatter], [legend_label])

    plt.tight_layout()
    plt.savefig(f"../output/tsne-{name}-{type}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to ../output/tsne-{name}-{type}.png")


def print_div_sem(
    df: pd.DataFrame,
    type: str,
    name: str,
    sample_size: int = 5000,
    n_repeats: int = 10,
):
    """Diversity metric from: https://aclanthology.org/2024.findings-naacl.228.pdf

    div_sem is the mean pairwise cosine distance over the embeddings. Estimating
    it from a single random sample is noisy: the value swings depending on which
    queries are drawn. To stabilise it we embed the whole pool *once* (the
    expensive step) and then average the metric over `n_repeats` random subsamples
    of `sample_size` queries drawn from that cached pool. Averaging shrinks the
    standard deviation of the estimate by ~sqrt(n_repeats), and we also report
    the std so a difference between datasets can be told apart from sampling noise.

    Args:
        df (pd.DataFrame): _description_
        type (str): _description_
        name (str): _description_
        sample_size (int): number of queries per subsample.
        n_repeats (int): number of subsamples to average over.
    """

    # A pairwise distance needs at least two points. The train splits are
    # normal-only (attacks live in the test split), so the attack pool is empty
    # there -- skip it cleanly (before the costly embedding step) instead of
    # crashing on a 0-row array.
    if len(df) < 2:
        print(
            f"Semantic Diversity of {type} for dataset {name}: skipped "
            f"(only {len(df)} queries available, need >= 2)."
        )
        return {
            "div_sem_mean": float("nan"),
            "div_sem_std": float("nan"),
            "div_sem_n": len(df),
        }

    _embeddings = compute_embeddings(df=df)
    n = len(_embeddings)

    # If the pool is no larger than the sample size there is nothing to resample,
    # so just compute the metric once on everything.
    if n <= sample_size:
        div_sem = float(np.mean(pdist(_embeddings, metric="cosine")))
        print(
            f"Semantic Diversity of {type} for dataset {name} using cosine distance "
            f"(n={n}, single pass): {div_sem}"
        )
        return {"div_sem_mean": div_sem, "div_sem_std": 0.0, "div_sem_n": n}

    # Fixed rng so the reported value is reproducible; averaging is what reduces
    # the variance across the (upstream) data seed.
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=sample_size, replace=False)
        values.append(np.mean(pdist(_embeddings[idx], metric="cosine")))

    values = np.array(values)
    print(
        f"Semantic Diversity of {type} for dataset {name} using cosine distance "
        f"({n_repeats}x{sample_size} from pool of {n}): "
        f"{values.mean():.6f} +/- {values.std():.6f}"
    )
    return {
        "div_sem_mean": float(values.mean()),
        "div_sem_std": float(values.std()),
        "div_sem_n": n,
    }


def load_wafamole_samples(fp_sane: str, fp_attacks: str):
    # This is too long to parse each time, let's also save them as pickles.
    fp_patks = "../output/parsed-wafamole-attacks.pkl"
    fp_psane = "../output/parsed-wafamole-sane.pkl"

    if os.path.isfile(fp_patks):
        attacks = pd.read_pickle(fp_patks)
    else:
        attack = open(fp_attacks, "r").read()
        attacks = sqlparse.split(attack)
        pd.to_pickle(attacks, fp_patks)

    if os.path.isfile(fp_psane):
        sanes = pd.read_pickle(fp_psane)
    else:
        sane = open(fp_sane, "r").read()
        sanes = sqlparse.split(sane)
        pd.to_pickle(sanes, fp_psane)

    df_sane = pd.DataFrame(sanes, columns=["full_query"])
    df_attack = pd.DataFrame(attacks, columns=["full_query"])

    df_sane = df_sane.assign(label=0)
    df_attack = df_attack.assign(label=1)

    return pd.concat([df_sane, df_attack])


def process_dataset(
    df: pd.DataFrame,
    name: str,
    query_column: str = "full_query",
    label_column: str = "label",
    split_column: str = "split",
    sem_sample_size: int = 20000,
    vocab: bool = True,
    parse_trees: bool = True,
    div_sem: bool = True,
) -> list:
    """Compute lexical (vocab), syntactic (parse trees) and semantic (div_sem)
    diversity for one dataset.

    Source layout (see experiments/build_big_trainsets.py): the *train* split is
    benign-only (~100k normal queries, label 0) and the attacks live in the
    *test* split (~100k, label 1). We therefore draw:
      - normal queries from the train split,
      - attack queries from the test split.

    Lexical and syntactic metrics are cheap enough to run on the *full* normal and
    attack pools. Semantic diversity (the expensive SecureBERT embedding step) is
    run on a random sub-pool of `sem_sample_size` queries per label.

    All three metrics default to on so a single call reports lex + synt + sem.

    Returns one row dict per label (normal, attack) with every computed metric, so
    the caller can aggregate the rows into the per-mode results CSV.
    """
    # (label type, source split, frame) for the two pools we measure.
    pools = [
        ("normal", "train", df[(df[split_column] == "train") & (df[label_column] == 0)]),
        ("attack", "test", df[(df[split_column] == "test") & (df[label_column] == 1)]),
    ]

    rows = []
    for qtype, src_split, pool in pools:
        queries = pool[query_column].tolist()
        print(f"[{name}] {qtype} pool ({src_split} split): {len(queries)} queries")

        row = {
            "dataset": name,
            "type": qtype,
            "source_split": src_split,
            "n_queries": len(queries),
        }

        # Lexical and syntactic metrics run on the full pool.
        if vocab:
            row.update(print_vocab_size(queries, qtype, name))
        if parse_trees:
            row.update(print_unique_pts(queries, qtype, name))

        # Semantic metric runs on a (capped) random sub-pool. print_div_sem
        # further resamples within this pool to stabilise the estimate.
        if div_sem:
            pool_sem = pool.rename(columns={query_column: "full_query"})
            if len(pool_sem) > sem_sample_size:
                pool_sem = pool_sem.sample(n=sem_sample_size, random_state=2)
            row.update(print_div_sem(pool_sem, qtype, name))

        rows.append(row)

    return rows


# In-domain (train and test from the same domain) and LODO (train on three
# domains, test on the held-out one) scenario -> filename maps.
INDOMAIN_DATASETS = {
    "A-A": "a-a.csv",
    "B-B": "b-b.csv",
    "C-C": "c-c.csv",
    "D-D": "d-d.csv",
}

LODO_DATASETS = {
    "BCD-A": "bcd-a.csv",
    "ACD-B": "acd-b.csv",
    "ABD-C": "abd-c.csv",
    "ABC-D": "abc-d.csv",
}


def process_datasets(datasets: dict, results_filename: str):
    """Compute lexical, syntactic and semantic diversity for each dataset and
    write one aggregated CSV (`results_filename`, under ../output) with all
    metrics, one row per (dataset, label type).

    For every CSV, process_dataset draws normal queries from the train split and
    attacks from the test split, computes lexical/syntactic metrics on the full
    pools and semantic diversity on a 20k sub-pool per label. In-domain and LODO
    differ only in which files are processed (see process_dataset for the why).
    """
    rows = []
    for name, filename in datasets.items():
        df = pd.read_csv(DATASETS_DIR / filename, dtype=DTYPES)
        rows.extend(process_dataset(df=df, name=name))

    results = pd.DataFrame(rows)
    out_path = Path("../output") / results_filename
    results.to_csv(out_path, index=False)
    print(f"Wrote aggregated metrics ({len(results)} rows) to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute lexical (vocab), syntactic (parse trees) and semantic "
            "(div_sem) diversity of the SQLIA datasets. Normal queries come from "
            "the train split, attacks from the test split; lexical/syntactic run "
            "on the full pools, semantic on a 20k sub-pool per label. Choose "
            "'indomain' for A->A, B->B, ... or 'lodo' for the cross-domain "
            "leave-one-domain-out sets BCD->A, ACD->B, ..."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["indomain", "lodo"],
        required=True,
        help="Which datasets to process.",
    )
    args = parser.parse_args()

    Path("../output").mkdir(exist_ok=True, parents=True)

    if args.mode == "indomain":
        process_datasets(INDOMAIN_DATASETS, "results_indomain.csv")
    elif args.mode == "lodo":
        process_datasets(LODO_DATASETS, "results_lodo.csv")


if __name__ == "__main__":
    main()
