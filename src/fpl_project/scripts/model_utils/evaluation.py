"""Evaluation of predictions on the original target scale, overall and per price tier."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_raw: np.ndarray,
    weights: np.ndarray,
    p_low: float,
    p_high: float,
    p_features: int,
) -> dict[str, float | int]:
    """Compute overall and price-tier metrics on original clipped target scale."""
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    price = np.asarray(price_raw, dtype=np.float64).flatten()
    w = np.asarray(weights, dtype=np.float64).flatten()

    n = len(y_t)
    err = y_p - y_t
    sq_err = err ** 2
    abs_err = np.abs(err)

    mse = float(np.mean(sq_err))
    mae = float(np.mean(abs_err))
    r2 = float(r2_score(y_t, y_p))
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p_features - 1)) if n > p_features + 1 else 0.0
    fplloss_val = float(np.mean(w * sq_err))

    mask_b = price < p_low
    mask_m = (price >= p_low) & (price < p_high)
    mask_p = price >= p_high

    return {
        "p": int(p_features),
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "fplloss": fplloss_val,
        "mse@budget": float(np.mean(sq_err[mask_b])) if mask_b.any() else 0.0,
        "mse@mid": float(np.mean(sq_err[mask_m])) if mask_m.any() else 0.0,
        "mse@premium": float(np.mean(sq_err[mask_p])) if mask_p.any() else 0.0,
        "mae@budget": float(np.mean(abs_err[mask_b])) if mask_b.any() else 0.0,
        "mae@mid": float(np.mean(abs_err[mask_m])) if mask_m.any() else 0.0,
        "mae@premium": float(np.mean(abs_err[mask_p])) if mask_p.any() else 0.0,
    }


def compute_tier_counts_table(
    splits_dict: dict[str, pd.DataFrame],
    p_low: float,
    p_high: float,
) -> pd.DataFrame:
    """Count observations in budget, mid, and premium tiers across splits."""
    rows = []
    for split_name, df_part in splits_dict.items():
        price = df_part["value_ema_3"].to_numpy(dtype=np.float64)
        rows.append({
            "split": split_name,
            "budget": int(np.sum(price < p_low)),
            "mid": int(np.sum((price >= p_low) & (price < p_high))),
            "premium": int(np.sum(price >= p_high)),
            "total": int(len(df_part)),
        })
    return pd.DataFrame(rows)
