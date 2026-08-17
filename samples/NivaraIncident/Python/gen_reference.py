"""Generate Polars reference outputs for C# cross-validation.

Produces three manifests:

1. samples/data/polars-window/manifest.json     — generic rank-family and rolling-window cases
2. samples/data/polars-quantile/manifest.json   — generic quantile and median cases
3. samples/data/polars-moments/manifest.json    — generic stddev and variance cases
4. samples/data/polars-incident/manifest.json   — incident-scenario fixtures (latency percentiles,
   error-rate rolling windows, rank/percentrank by error delta, stddev per service)

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
    the changed manifest.json files as one unit.

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

    incident_manifest = emit_incident_fixtures(pl)
    print(f"\nTotal incident test cases: {len(incident_manifest)}")


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


# ---------------------------------------------------------------------------
# Incident-specific fixtures: realistic telemetry distributions per service.
# ---------------------------------------------------------------------------

# Latency percentiles per service: (name, service_name -> values, q).
# Values are in milliseconds; hand-authored to look like realistic service latency
# distributions (gateway fast, payments slow, notifications bimodal).  Some nulls.
INCIDENT_QUANTILE_CASES = [
    ("latency_p50_per_service", {
        "gateway":       [12.1, 8.3, 15.7, None, 11.2, 9.8, 14.3, 7.6, 13.5, 10.1, 11.9, 8.7, 16.2, None, 10.8],
        "orders":        [45.2, 38.7, 52.1, 41.3, None, 48.6, 35.9, 55.8, 42.0, 39.4, 51.2, 44.8, None, 47.3, 37.5],
        "checkout":      [78.4, 65.2, 92.1, 71.8, 85.3, None, 69.5, 88.7, 74.6, 81.2, 95.3, 62.8, 77.9, 83.1, None],
        "payments":      [120.5, 98.3, 145.2, 112.7, 133.8, 105.6, None, 141.9, 118.4, 127.3, 155.1, 108.2, 136.7, 115.8, 142.6],
        "notifications": [25.3, 18.7, None, 31.2, 22.8, 45.6, 20.1, 28.9, None, 33.4, 19.5, 42.1, 27.6, 35.8, 21.4],
    }, 0.5),
    ("latency_p95_per_service", {
        "gateway":       [12.1, 8.3, 15.7, None, 11.2, 9.8, 14.3, 7.6, 13.5, 10.1, 11.9, 8.7, 16.2, None, 10.8],
        "orders":        [45.2, 38.7, 52.1, 41.3, None, 48.6, 35.9, 55.8, 42.0, 39.4, 51.2, 44.8, None, 47.3, 37.5],
        "checkout":      [78.4, 65.2, 92.1, 71.8, 85.3, None, 69.5, 88.7, 74.6, 81.2, 95.3, 62.8, 77.9, 83.1, None],
        "payments":      [120.5, 98.3, 145.2, 112.7, 133.8, 105.6, None, 141.9, 118.4, 127.3, 155.1, 108.2, 136.7, 115.8, 142.6],
        "notifications": [25.3, 18.7, None, 31.2, 22.8, 45.6, 20.1, 28.9, None, 33.4, 19.5, 42.1, 27.6, 35.8, 21.4],
    }, 0.95),
    ("latency_p99_per_service", {
        "gateway":       [12.1, 8.3, 15.7, None, 11.2, 9.8, 14.3, 7.6, 13.5, 10.1, 11.9, 8.7, 16.2, None, 10.8],
        "orders":        [45.2, 38.7, 52.1, 41.3, None, 48.6, 35.9, 55.8, 42.0, 39.4, 51.2, 44.8, None, 47.3, 37.5],
        "checkout":      [78.4, 65.2, 92.1, 71.8, 85.3, None, 69.5, 88.7, 74.6, 81.2, 95.3, 62.8, 77.9, 83.1, None],
        "payments":      [120.5, 98.3, 145.2, 112.7, 133.8, 105.6, None, 141.9, 118.4, 127.3, 155.1, 108.2, 136.7, 115.8, 142.6],
        "notifications": [25.3, 18.7, None, 31.2, 22.8, 45.6, 20.1, 28.9, None, 33.4, 19.5, 42.1, 27.6, 35.8, 21.4],
    }, 0.99),
]

# Error-rate rolling windows per service: (name, service -> values, window_size, min_samples).
# Values are error-rate percentages per time interval; some nulls simulate missing intervals.
INCIDENT_ROLLING_CASES = [
    ("error_rate_rolling_window_per_service", {
        "gateway":    [0.1, 0.3, None, 0.2, 1.8, 5.2, 12.3, 8.7, 3.1, 1.2, 0.8, 0.4],
        "orders":     [0.5, 0.4, 0.6, 0.3, 2.1, 8.5, 15.7, 22.3, 18.1, 9.2, 4.3, 2.1],
        "checkout":   [0.2, 0.1, 0.3, None, 1.5, 6.8, 11.4, 16.9, 12.5, 7.1, 3.8, 1.5],
        "payments":   [1.2, 0.9, 1.1, 0.8, 3.4, 12.6, 25.8, 31.2, 28.5, 15.3, 8.7, 4.2],
    }, 3, 1),
]

# Rank/PercentRank per service by error delta: (name, service -> error_delta).
# Error deltas are percentage-point increases.  Null means no data for that service.
INCIDENT_RANK_CASES = [
    ("rank_by_error_delta", {
        "gateway":       418,
        "orders":        172,
        "checkout":      91,
        "payments":      312,
        "notifications": None,
        "catalog":       4,
    }),
    ("rank_by_error_delta_with_ties", {
        "gateway":       150,
        "orders":        91,
        "checkout":      91,
        "payments":      200,
        "notifications": 50,
        "catalog":       4,
    }),
]

# StdDev per service: (name, service -> values, ddof).  Latency values in ms.
INCIDENT_STDDEV_CASES = [
    ("stddev_latency_per_service", {
        "gateway":       [12.1, 8.3, 15.7, 11.2, 9.8, 14.3, 7.6, 13.5, 10.1, 11.9, 8.7, 16.2, 10.8],
        "orders":        [45.2, 38.7, 52.1, 41.3, 48.6, 35.9, 55.8, 42.0, 39.4, 51.2, 44.8, 47.3, 37.5],
        "checkout":      [78.4, 65.2, 92.1, 71.8, 85.3, 69.5, 88.7, 74.6, 81.2, 95.3, 62.8, 77.9, 83.1],
        "payments":      [120.5, 98.3, 145.2, 112.7, 133.8, 105.6, 141.9, 118.4, 127.3, 155.1, 108.2, 136.7, 115.8],
    }, 0),
    ("stddev_sample_latency_per_service", {
        "gateway":       [12.1, 8.3, 15.7, 11.2, 9.8, 14.3, 7.6, 13.5, 10.1, 11.9, 8.7, 16.2, 10.8],
        "orders":        [45.2, 38.7, 52.1, 41.3, 48.6, 35.9, 55.8, 42.0, 39.4, 51.2, 44.8, 47.3, 37.5],
        "checkout":      [78.4, 65.2, 92.1, 71.8, 85.3, 69.5, 88.7, 74.6, 81.2, 95.3, 62.8, 77.9, 83.1],
        "payments":      [120.5, 98.3, 145.2, 112.7, 133.8, 105.6, 141.9, 118.4, 127.3, 155.1, 108.2, 136.7, 115.8],
    }, 1),
]


def emit_incident_fixtures(pl):
    """Emit incident-scenario fixtures to samples/data/polars-incident/.

    These use realistic telemetry distributions (latency in ms, error rates as
    percentages) across multiple services, exercising the same Nivara APIs as the
    generic fixtures but on plausible production-like data.
    """
    incident_dir = os.path.join(REPO_ROOT, "samples", "data", "polars-incident")
    os.makedirs(incident_dir, exist_ok=True)
    incident_manifest = []

    # --- Latency percentiles per service ---
    for name, services, q in INCIDENT_QUANTILE_CASES:
        expected = {}
        for svc, values in services.items():
            series = pl.Series("v", values, dtype=pl.Float64)
            val = series.quantile(q, interpolation="linear")
            expected[svc] = None if val is None else float(val)
        case = {
            "name": name,
            "kind": "quantile_per_service",
            "q": q,
            "services": {svc: [None if v is None else float(v) for v in vals]
                         for svc, vals in services.items()},
            "expected": expected,
        }
        incident_manifest.append(case)
        print(f"  {name}: q={q} expected={expected}")

    # --- Error-rate rolling windows per service ---
    for name, services, window_size, min_samples in INCIDENT_ROLLING_CASES:
        expected_rolling_mean = {}
        for svc, values in services.items():
            df = pl.DataFrame({"g": [svc] * len(values), "v": values})
            result = df.with_columns(
                pl.col("v").rolling_mean(window_size=window_size, min_samples=min_samples).alias("rolling_mean")
            )
            means = [None if v is None else float(v)
                     for v in result["rolling_mean"].to_list()]
            expected_rolling_mean[svc] = means
        case = {
            "name": name,
            "kind": "rolling_per_service",
            "window_size": window_size,
            "min_samples": min_samples,
            "services": {svc: [None if v is None else float(v) for v in vals]
                         for svc, vals in services.items()},
            "expected_rolling_mean": expected_rolling_mean,
        }
        incident_manifest.append(case)
        print(f"  {name}: window={window_size} min_s={min_samples} services={list(services.keys())}")

    # --- Rank/PercentRank per service by error delta ---
    for name, services in INCIDENT_RANK_CASES:
        # Build a dataframe with non-null services for rank computation
        svc_names = []
        deltas = []
        for svc, delta in services.items():
            if delta is not None:
                svc_names.append(svc)
                deltas.append(delta)

        df = pl.DataFrame({"service": svc_names, "delta": deltas})
        ranked = df.with_columns([
            pl.col("delta").rank(method="min", descending=True).alias("rank"),
            pl.col("delta").rank(method="dense", descending=True).alias("dense_rank"),
            ((pl.col("delta").rank(method="min", descending=True) - 1)
             / (pl.len() - 1)).alias("percent_rank"),
        ])

        # Build expected dicts (only non-null services get ranks; null services -> null)
        expected_rank = {}
        expected_dense = {}
        expected_pct = {}
        for row in ranked.iter_rows(named=True):
            expected_rank[row["service"]] = int(row["rank"])
            expected_dense[row["service"]] = int(row["dense_rank"])
            expected_pct[row["service"]] = float(row["percent_rank"])
        for svc in services:
            if services[svc] is None:
                expected_rank[svc] = None
                expected_dense[svc] = None
                expected_pct[svc] = None

        case = {
            "name": name,
            "kind": "rank_per_service",
            "services": {svc: delta for svc, delta in services.items()},
            "expected_rank": expected_rank,
            "expected_dense_rank": expected_dense,
            "expected_percent_rank": expected_pct,
        }
        incident_manifest.append(case)
        print(f"  {name}: rank={expected_rank} dense={expected_dense} pct={expected_pct}")

    # --- StdDev per service ---
    for name, services, ddof in INCIDENT_STDDEV_CASES:
        expected = {}
        for svc, values in services.items():
            series = pl.Series("v", values, dtype=pl.Float64)
            val = series.std(ddof=ddof)
            expected[svc] = None if val is None else float(val)
        case = {
            "name": name,
            "kind": "stddev_per_service",
            "ddof": ddof,
            "services": {svc: [None if v is None else float(v) for v in vals]
                         for svc, vals in services.items()},
            "expected": expected,
        }
        incident_manifest.append(case)
        print(f"  {name}: ddof={ddof} expected={expected}")

    manifest_path = os.path.join(incident_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(incident_manifest, f, indent=2)
    print(f"\nIncident manifest: {manifest_path}")
    print(f"Total incident test cases: {len(incident_manifest)}")
    return incident_manifest


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
