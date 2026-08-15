"""Generate Polars rank-family reference outputs for C# cross-validation.

Saves a JSON manifest with fixed input arrays (order keys with nulls, optional
partition labels) and the expected row_number / rank / dense_rank / percent_rank
outputs per case. Test fixtures go to samples/data/polars-window/.

Semantics being pinned:
  - row_number   -> polars rank(method="ordinal")
  - rank         -> polars rank(method="min")          (standard rank, gaps on ties)
  - dense_rank   -> polars rank(method="dense")
  - percent_rank -> (rank - 1) / (nonNullCount - 1)    (null keys -> null)
  - Polars ranks null-key rows as null for every method.

Nivara divergence (documented, not part of the fixture contract): row_number
numbers null-key rows LAST in stable partition order (issue #254, SQL semantics)
where Polars emits null. The C# test asserts Polars parity on non-null-key rows
and Nivara's documented numbered-last values on null-key rows.

Reproducibility:
  - Verified with Python 3.12, polars 1.43.2.
  - All cases are hand-authored fixed arrays; no RNG, so output is stable across
    polars versions as long as the rank semantics above hold.
  - Regenerate after upgrading polars: run `python gen_reference.py` and commit
    samples/data/polars-window/manifest.json as one unit.

Usage: python gen_reference.py
"""
import os
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_DIR = os.path.join(REPO_ROOT, "samples", "data", "polars-window")

# Case tuples: (name, partition labels or None, order keys with nulls).
# Every partition holds >= 2 valid rows so percent_rank never divides by zero.
CASES = [
    ("row_number_sorted_global", None, [2, 1, 3]),
    ("row_number_partitioned", ["A", "A", "B", "B", "A"], [3, 1, 2, 2, 0]),
    ("row_number_null_keys", None, [2, None, 1, None]),
    ("rank_with_nulls", None, [2, None, 1, None]),
    ("dense_rank_with_nulls_and_ties", None, [3, None, 1, 1]),
    ("percent_rank_with_nulls", None, [4, 2, None, 2, 1]),
    ("rank_partitioned_with_nulls", ["X", "X", "Y", "Y", "X"], [5, None, 2, 2, 1]),
]


def run():
    import polars as pl

    os.makedirs(TEST_DIR, exist_ok=True)
    print(f"polars {pl.__version__}")

    manifest = []
    for name, partition, order in CASES:
        df = pl.DataFrame(
            {
                "g": partition if partition is not None else ["G"] * len(order),
                "v": order,
            }
        )
        group = [] if partition is not None else None
        rank_exprs = [
            pl.col("v").rank(method="ordinal").over("g").alias("row_number"),
            pl.col("v").rank(method="min").over("g").alias("rank"),
            pl.col("v").rank(method="dense").over("g").alias("dense_rank"),
            (
                (pl.col("v").rank(method="min").over("g") - 1)
                / (pl.col("v").count().over("g") - 1)
            ).alias("percent_rank"),
        ]
        out = df.with_columns(rank_exprs)

        def col_values(cname):
            series = out[cname]
            return [
                None if v is None else v
                for v in series.to_list()
            ]

        case = {
            "name": name,
            "partition": partition,
            "order": order,
            "row_number": col_values("row_number"),
            "rank": col_values("rank"),
            "dense_rank": col_values("dense_rank"),
            "percent_rank": col_values("percent_rank"),
        }
        manifest.append(case)
        print(
            f"  {name}: rn={case['row_number']} rank={case['rank']} "
            f"dense={case['dense_rank']} pct={case['percent_rank']}"
        )

    manifest_path = os.path.join(TEST_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Total test cases: {len(manifest)}")


if __name__ == "__main__":
    run()
