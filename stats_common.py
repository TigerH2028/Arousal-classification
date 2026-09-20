"""
FILE: stats_common.py
PURPOSE: Single source of truth for the statistics shared across Table 4
         (generate_table1.py / compute_table1_stats.py) and Table 5
         (univariate_logistic_regression.py). Both tables now import from
         here rather than reimplementing these formulas locally, so they
         cannot numerically drift apart again.

         Also provides complete_case_sample(), the ONE sample-selection rule
         used everywhere in the paper: a window is dropped only if at least
         one of the 15 features is missing (listwise deletion across all
         features at once), matching the sample used in the classification
         pipeline (tuned_classification_models.py, Table 6/7).
"""

import numpy as np
from scipy import stats

BONFERRONI_N = 15
BONFERRONI_ALPHA = 0.05 / BONFERRONI_N


def cohens_d_pooled(high, low):
    """Cohen's d using the pooled-SD formula (paper Eq. 3-4)."""
    n1, n2 = len(high), len(low)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.std(high, ddof=1), np.std(low, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_sd <= 1e-10:
        return np.nan
    return (np.mean(high) - np.mean(low)) / pooled_sd


def welch_ttest(high, low):
    """Welch's t-test (paper Eq. 1-2). Returns (t_stat, welch_satterthwaite_df, p_value)."""
    n1, n2 = len(high), len(low)
    t_stat, p_val = stats.ttest_ind(high, low, equal_var=False)
    se1_sq = np.var(high, ddof=1) / n1
    se2_sq = np.var(low, ddof=1) / n2
    df_w = (se1_sq + se2_sq) ** 2 / (
        se1_sq ** 2 / (n1 - 1) + se2_sq ** 2 / (n2 - 1)
    )
    return t_stat, df_w, p_val


def r_pb_exact(d, n1, n2):
    """Exact point-biserial r from Cohen's d (Cohen 1988, p.24; paper Eq. 6)."""
    return d / np.sqrt(d ** 2 + (n1 + n2) ** 2 / (n1 * n2))


def effect_size_zone(value, kind):
    """kind = 'd' or 'r'. Thresholds from Cohen (1988)."""
    v = abs(value)
    if kind == 'd':
        if v >= 0.80:
            return 'Large'
        if v >= 0.50:
            return 'Medium'
        if v >= 0.20:
            return 'Small'
        return 'Negligible'
    else:  # r_pb
        if v >= 0.50:
            return 'Large'
        if v >= 0.30:
            return 'Medium'
        if v >= 0.10:
            return 'Small'
        return 'Negligible'


def sig_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'NS'


def complete_case_sample(df, feature_cols, label_col='arousal_binary'):
    """
    Listwise deletion across ALL feature_cols at once -- the shared sample
    used by Table 4, Table 5, and the classification models (Table 6/7),
    so every table in the paper is computed on the identical N.

    Returns (clean_df, avail_feature_cols).
    """
    avail = [f for f in feature_cols if f in df.columns]
    clean = df[avail + [label_col]].dropna()
    return clean, avail
