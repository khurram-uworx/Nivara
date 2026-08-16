"""Generate Polars rank-family and rolling-window reference outputs for C# cross-validation.

Saves a JSON manifest with fixed input arrays (order keys with nulls, optional
partition labels) and the expected row_number / rank / dense_rank / percent_rank
outputs, plus rolling sum / mean / min / max per case. Test fixtures go to
samples/data/polars-window/.

Semantics being pinned:
  - row_number   -> polars rank(method="ordinal")
  - rank         -> polars rank(method="min")          (standard rank, gaps on ties)
  - dense_rank   -> polars rank(method="dense")
  - percent_rank -> (rank - 1) / (nonNullCount - 1)    (null keys -> null)
  - Polars ranks null-key rows as null for every method.
  - rolling_*    -> polars rolling_sum/mean/min/max with window_size and
    min_samples. Polars skips null values inside the window and emits null when
    the non-null count in the window is below min_samples — matching Nivara's
    minPeriods gating. Values are authored in rolling (row) order; partitioned
    rolling is confined to each group (over("g")).

Nivara divergence (documented, not part of the fixture contract): row_number
numbers null-key rows LAST in stable partition order (issue #254, SQL semantics)
where Polars emits null. The C# test asserts Polars parity on non-null-key rows
and Nivara's documented numbered-last values on null-key rows.

Reproducibility:
  - Verified with Python 3.12, polars 1.43.2.
  - All cases are hand-authored fixed arrays; no RNG, so output is stable across
    polars versions as long as the semantics above hold.
  - Regenerate after upgrading polars: run `python gen_reference.py` and commit
    samples/data/polars-window/manifest.json as one unit.

Usage: python gen_reference.py
"""
import os
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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

# Rolling case tuples: (name, partition labels or None, values with nulls,
# window_size, min_samples). Values are authored in rolling (row) order; the
# window is confined to each partition. min_samples <= window_size always.
ROLLING_CASES = [
    ("rolling_sum_global_default", None, [2, 4, 1, 3], 2, 2),
    ("rolling_mean_global_min1", None, [1, 3, None, 2], 3, 1),
    ("rolling_min_global_nulls", None, [5, None, 3, 1, 4], 3, 2),
    ("rolling_max_global_nulls_default", None, [2, None, 8, 1, None, 5], 3, 3),
    ("rolling_sum_partitioned_min1_nulls", ["A", "A", "B", "B", "A", "B"], [1, None, 3, None, 5, 6], 2, 1),
    ("rolling_mean_partitioned_default_nulls", ["X", "X", "Y", "X", "Y", "Y"], [4, 6, None, 2, None, 8], 2, 2),
    ("rolling_min_partitioned_ties_min2", ["P", "P", "Q", "P", "Q"], [3, 3, 5, 3, 3], 2, 2),
    ("rolling_max_partitioned_windowLargerThanGroup_min1", ["A", "A", "B"], [1, 2, 7], 5, 1),
]

# Quantile cases: (name, values with nulls, q). Values are authored in row order;
# the fixture pins polars quantile(q, interpolation="linear") over the non-null values.
QUANTILE_CASES = [
    ("quantile_linear_basic", [2, 4, 1, 3], 0.5),
    ("quantile_linear_odd", [10, 20, 30, 40, 50], 0.9),
    ("quantile_linear_p95", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95),
    ("quantile_linear_nulls", [5, None, 3, 1, 4], 0.25),
    ("quantile_linear_single", [7], 0.5),
    ("quantile_linear_min", [3, 9, 6], 0.0),
    ("quantile_linear_max", [3, 9, 6], 1.0),
    ("quantile_linear_even_median", [100, 200, 300, 400], 0.5),
]

# Median cases: (name, partition labels or None, values with nulls). Per-group values are
# pinned via polars median().over("g") when partitioned, plus the whole-column median.
MEDIAN_CASES = [
    ("median_global_odd", None, [3, 1, 2]),
    ("median_global_even", None, [1, 3, 2, 4]),
    ("median_global_nulls", None, [5, None, 3, 1, 4]),
    ("median_partitioned", ["A", "A", "B", "B", "A", "B"], [10, 30, 2, 8, 20, 6]),
]

# StdDev/Variance cases: (name, kind, values with nulls, ddof). Pinned against numpy
# np.std/np.var (equivalently polars Series.std/var with ddof), so ddof=0 is population
# (divide by n) and ddof=1 is sample (divide by n-1). Every case keeps at least ddof+1
# non-null values so the divisor is positive.
MOMENT_CASES = [
    ("std_pop_basic", "stddev", [2, 4, 4, 4, 5, 5, 7, 9], 0),
    ("std_sample_basic", "stddev", [2, 4, 4, 4, 5, 5, 7, 9], 1),
    ("std_single_pop", "stddev", [7], 0),
    ("std_nulls", "stddev", [5, None, 3, 1, 4], 0),
    ("std_constant", "stddev", [3, 3, 3], 0),
    ("var_pop_basic", "variance", [2, 4, 4, 4, 5, 5, 7, 9], 0),
    ("var_sample_basic", "variance", [2, 4, 4, 4, 5, 5, 7, 9], 1),
    ("var_nulls", "variance", [5, None, 3, 1, 4], 0),
]


def col_values(out, cname):
    """Polars column to a Python list, preserving nulls."""
    series = out[cname]
    return [None if v is None else v for v in series.to_list()]


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

        case = {
            "name": name,
            "kind": "rank",
            "partition": partition,
            "order": order,
            "row_number": col_values(out, "row_number"),
            "rank": col_values(out, "rank"),
            "dense_rank": col_values(out, "dense_rank"),
            "percent_rank": col_values(out, "percent_rank"),
        }
        manifest.append(case)
        print(
            f"  {name}: rn={case['row_number']} rank={case['rank']} "
            f"dense={case['dense_rank']} pct={case['percent_rank']}"
        )

    for name, partition, values, window_size, min_samples in ROLLING_CASES:
        df = pl.DataFrame(
            {
                "g": partition if partition is not None else ["G"] * len(values),
                "v": values,
            }
        )
        rolling_exprs = [
            pl.col("v").rolling_sum(window_size=window_size, min_samples=min_samples).over("g").alias("rolling_sum"),
            pl.col("v").rolling_mean(window_size=window_size, min_samples=min_samples).over("g").alias("rolling_mean"),
            pl.col("v").rolling_min(window_size=window_size, min_samples=min_samples).over("g").alias("rolling_min"),
            pl.col("v").rolling_max(window_size=window_size, min_samples=min_samples).over("g").alias("rolling_max"),
        ]
        out = df.with_columns(rolling_exprs)

        case = {
            "name": name,
            "kind": "rolling",
            "partition": partition,
            "v": [None if v is None else v for v in values],
            "window_size": window_size,
            "min_samples": min_samples,
            "rolling_sum": col_values(out, "rolling_sum"),
            "rolling_mean": col_values(out, "rolling_mean"),
            "rolling_min": col_values(out, "rolling_min"),
            "rolling_max": col_values(out, "rolling_max"),
        }
        manifest.append(case)
        print(
            f"  {name}: sum={case['rolling_sum']} mean={case['rolling_mean']} "
            f"min={case['rolling_min']} max={case['rolling_max']}"
        )

    manifest_path = os.path.join(TEST_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Total test cases: {len(manifest)}")

    quantile_manifest = emit_quantile_fixtures(pl)
    print(f"\nTotal quantile/median test cases: {len(quantile_manifest)}")

    moments_manifest = emit_moment_fixtures(pl)
    print(f"\nTotal stddev/variance test cases: {len(moments_manifest)}")


def emit_quantile_fixtures(pl):
    """Emit polars quantile(linear) and median fixtures to samples/data/polars-quantile/."""
    quantile_dir = os.path.join(REPO_ROOT, "samples", "data", "polars-quantile")
    os.makedirs(quantile_dir, exist_ok=True)
    quantile_manifest = []

    for name, values, q in QUANTILE_CASES:
        series = pl.Series("v", values, dtype=pl.Float64)
        quantile = series.quantile(q, interpolation="linear")
        median = series.median()
        if abs(q - 0.5) < 1e-12:
            assert abs(quantile - median) < 1e-12, (
                f"{name}: linear q=0.5 quantile {quantile} must match median {median}"
            )
        case = {
            "name": name,
            "kind": "quantile",
            "v": [None if v is None else float(v) for v in values],
            "q": q,
            "quantile": None if quantile is None else float(quantile),
        }
        quantile_manifest.append(case)
        print(f"  {name}: q={q} quantile={case['quantile']}")

    for name, partition, values in MEDIAN_CASES:
        df = pl.DataFrame(
            {
                "g": partition if partition is not None else ["G"] * len(values),
                "v": values,
            }
        )
        whole_median = df.select(pl.col("v").median()).to_series()[0]
        case = {
            "name": name,
            "kind": "median",
            "partition": partition,
            "v": [None if v is None else float(v) for v in values],
            "median": None if whole_median is None else float(whole_median),
        }
        if partition is not None:
            grouped = df.group_by("g").agg(pl.col("v").median().alias("median")).sort("g")
            case["groups"] = {
                str(row["g"]): (None if row["median"] is None else float(row["median"]))
                for row in grouped.iter_rows(named=True)
            }
        quantile_manifest.append(case)
        print(f"  {name}: median={case['median']}" + (f" groups={case['groups']}" if "groups" in case else ""))

    manifest_path = os.path.join(quantile_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(quantile_manifest, f, indent=2)
    print(f"\nQuantile manifest: {manifest_path}")
    return quantile_manifest


def emit_moment_fixtures(pl):
    """Emit numpy/polars stddev & variance fixtures to samples/data/polars-moments/.

    polars Series.std/var with an explicit ddof are numerically identical to numpy
    np.std/np.var (same sum-of-squared-deviation over n-ddof definition), so this
    doubles as the NumPy parity fixture from the plan. ddof=0 is population, ddof=1
    is sample. Nulls are ignored.
    """
    moments_dir = os.path.join(REPO_ROOT, "samples", "data", "polars-moments")
    os.makedirs(moments_dir, exist_ok=True)
    moments_manifest = []

    for name, kind, values, ddof in MOMENT_CASES:
        series = pl.Series("v", values, dtype=pl.Float64)
        if kind == "stddev":
            value = series.std(ddof=ddof)
        else:
            value = series.var(ddof=ddof)
        case = {
            "name": name,
            "kind": kind,
            "v": [None if v is None else float(v) for v in values],
            "ddof": ddof,
            "value": None if value is None else float(value),
        }
        moments_manifest.append(case)
        print(f"  {name}: {kind} ddof={ddof} value={case['value']}")

    manifest_path = os.path.join(moments_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(moments_manifest, f, indent=2)
    print(f"\nMoments manifest: {manifest_path}")
    return moments_manifest


if __name__ == "__main__":
    run()
