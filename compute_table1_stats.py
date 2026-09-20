"""
Computes the complete Table 1 (feature-level statistical significance table)
directly from raw per-window feature data.

CHANGED: now imports its statistics from stats_common.py and uses the SAME
complete-case sample (listwise deletion across all 15 features) as
univariate_logistic_regression.py (Table 5) and tuned_classification_models.py
(Table 6/7), instead of dropping NaNs one feature at a time. Table 4 and
Table 5 will now report identical Cohen's d / p-values for every feature.

USAGE:
    python compute_table1_stats.py --input features.csv --label arousal_label \
        --features gsr_min gsr_mean gsr_max ... hr_min hr_max hr_std rr_mean hrv_pnn50 \
        --output table1_all_emotions.csv

    Or import compute_table1() directly in your existing pipeline script.
"""

import argparse
import numpy as np
import pandas as pd

from stats_common import (
    cohens_d_pooled, welch_ttest, r_pb_exact, effect_size_zone,
    complete_case_sample, BONFERRONI_ALPHA,
)


def compute_table1(df: pd.DataFrame, feature_cols, label_col: str) -> pd.DataFrame:
    """
    df: raw per-window feature table (one row per window)
    feature_cols: list of feature column names (e.g. the 15 GSR/HRV features)
    label_col: binary arousal label column (1 = high, 0 = low)

    Returns a DataFrame with one row per feature and every Table 1 column,
    computed on the shared complete-case sample (see stats_common.complete_case_sample).
    """
    clean, avail = complete_case_sample(df, feature_cols, label_col=label_col)
    n_dropped = len(df) - len(clean)
    print(f"Complete-case sample: N={len(clean)} "
          f"(dropped {n_dropped} of {len(df)} windows with >=1 missing feature)")

    x_high_all = clean[clean[label_col] == 1]
    x_low_all = clean[clean[label_col] == 0]

    rows = []
    for feat in avail:
        x_high = x_high_all[feat].values
        x_low = x_low_all[feat].values

        n1, n2 = len(x_high), len(x_low)
        if n1 < 2 or n2 < 2:
            continue  # not enough data to compute variance

        high_mean, high_sd = np.mean(x_high), np.std(x_high, ddof=1)
        low_mean, low_sd = np.mean(x_low), np.std(x_low, ddof=1)

        t_stat, df_welch, p_val = welch_ttest(x_high, x_low)
        d = cohens_d_pooled(x_high, x_low)
        r = r_pb_exact(d, n1, n2)

        rows.append(
            {
                "Feature": feat,
                "High Mean (SD)": f"{high_mean:.3f} ({high_sd:.3f})",
                "Low Mean (SD)": f"{low_mean:.3f} ({low_sd:.3f})",
                "t": round(t_stat, 3),
                "df": round(df_welch, 1),
                "p": p_val,
                "p<.05": "YES" if p_val < 0.05 else "no",
                "Bonferroni sig (p<{:.4f})".format(BONFERRONI_ALPHA): (
                    "YES" if p_val < BONFERRONI_ALPHA else "no"
                ),
                "Cohen's d": round(d, 3),
                "d zone": effect_size_zone(d, "d"),
                "r_pb": round(r, 3),
                "r_pb zone": effect_size_zone(r, "r"),
                "n_high": n1,
                "n_low": n2,
            }
        )

    table = pd.DataFrame(rows)
    table["_abs_d"] = table["Cohen's d"].abs()
    table = table.sort_values("_abs_d", ascending=False).drop(columns="_abs_d")
    return table.reset_index(drop=True)


def format_p(p):
    """Match the report's scientific-notation style for small p-values."""
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV of raw per-window features")
    parser.add_argument("--label", required=True, help="Binary arousal label column")
    parser.add_argument("--features", nargs="+", required=True, help="Feature column names")
    parser.add_argument("--output", required=True, help="Output CSV path for Table 1")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    table1 = compute_table1(data, args.features, args.label)
    table1["p"] = table1["p"].apply(format_p)
    table1.to_csv(args.output, index=False)
    print(table1.to_string(index=False))
    print(f"\nSaved to {args.output}")
