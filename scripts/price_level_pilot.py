"""Validation-only pilot: price level vs reconstructed price.

The script never touches a test index.  For every fold it fits on the
selection-training rows and scores on the validation year only.  The
no-prediction forecast is the last observed adjusted close, ``P_t``.

Three families are compared:

* ``level``: fit ``P_{t+1}`` directly, with an unclipped price anchor added
  to the 13 stationary features (``log`` or ``raw``).
* ``resid``: fit ``P_{t+1} - P_t`` on the 13 features and add ``P_t`` back.
* ``logret``: fit ``log(P_{t+1}/P_t)`` on the 13 features and rebuild
  ``P_t * exp(prediction)``.

The last two produce a price forecast without asking a model to extrapolate
the level itself.  They are the economically sane way to "predict price".

Run:
    python scripts/price_level_pilot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURES  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.features import compute_features, fit_scaler, transform_scaler  # noqa: E402
from src.models.baselines import XGBModel  # noqa: E402
from src.models.denfis import DENFIS  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "n_estimators": 200,
}
LEVEL_MODELS = ["ridge", "denfis", "xgboost"]
RESIDUAL_MODELS = ["xgboost_resid", "denfis_resid"]
LOGRET_MODELS = ["xgboost_logret", "denfis_logret"]
ALL_MODELS = ["drift"] + LEVEL_MODELS + RESIDUAL_MODELS + LOGRET_MODELS


def _load() -> pd.DataFrame:
    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    price = prices.set_index("Date")["Adj Close"].astype(float)

    frame = compute_features(prices)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["price_now"] = frame["Date"].map(price)
    frame["price_next"] = frame["label_date"].map(price)
    frame = frame.dropna(subset=["price_now", "price_next"]).reset_index(drop=True)
    return frame


def _design(frame: pd.DataFrame, train_idx, val_idx, anchor: str):
    """Return the 13-feature design and the 13+price-anchor design."""
    scaler = fit_scaler(frame[FEATURES].iloc[train_idx])
    x13_train = transform_scaler(frame[FEATURES].iloc[train_idx], scaler)
    x13_val = transform_scaler(frame[FEATURES].iloc[val_idx], scaler)

    if anchor == "log":
        values = np.log(frame["price_now"].to_numpy())
    elif anchor == "raw":
        values = frame["price_now"].to_numpy()
    else:
        raise ValueError(anchor)
    train_values = values[train_idx]
    mean, std = train_values.mean(), train_values.std(ddof=1)
    if std == 0:
        std = 1.0
    xp_train = np.column_stack([x13_train, (train_values - mean) / std])
    xp_val = np.column_stack([x13_val, (values[val_idx] - mean) / std])
    return x13_train, x13_val, xp_train, xp_val


def _fit_predict(name, x_train, y_train, x_val, price_train, price_val):
    if name in ("ridge", "denfis", "xgboost"):
        target = y_train
        if name == "ridge":
            model = Ridge(alpha=1e3)
        elif name == "denfis":
            model = DENFIS(dthr=1.5, lam=1.0, prediction_clip=None, seed=42)
        else:
            model = XGBModel(XGB_PARAMS, seed=42)
        model.fit(x_train, target)
        return model.predict(x_val)

    if name.endswith("_resid"):
        target = y_train - price_train
        model = (
            XGBModel(XGB_PARAMS, seed=42)
            if name.startswith("xgboost")
            else DENFIS(dthr=1.5, lam=1.0, prediction_clip=None, seed=42)
        )
        model.fit(x_train, target)
        return price_val + model.predict(x_val)

    if name.endswith("_logret"):
        target = np.log(y_train / price_train)
        model = (
            XGBModel(XGB_PARAMS, seed=42)
            if name.startswith("xgboost")
            else DENFIS(dthr=1.5, lam=1.0, prediction_clip=None, seed=42)
        )
        model.fit(x_train, target)
        return price_val * np.exp(model.predict(x_val))

    raise ValueError(name)


def main() -> None:
    frame = _load()
    folds = [f for f in make_folds_from_frame(frame) if f.fold_id != "holdout"]
    rows = []

    for anchor in ("log", "raw"):
        for fold in folds:
            train_idx, val_idx = fold.select_train_idx, fold.val_idx
            x13_train, x13_val, xp_train, xp_val = _design(
                frame, train_idx, val_idx, anchor
            )

            y_train = frame["price_next"].iloc[train_idx].to_numpy()
            y_val = frame["price_next"].iloc[val_idx].to_numpy()
            price_train = frame["price_now"].iloc[train_idx].to_numpy()
            price_val = frame["price_now"].iloc[val_idx].to_numpy()

            naive_rmse = float(np.sqrt(np.mean((y_val - price_val) ** 2)))
            train_max = float(price_train.max())
            above = float(np.mean(price_val > train_max))
            actual_return = y_val / price_val - 1.0
            return_zero_rmse = float(np.sqrt(np.mean(actual_return**2)))

            simple_train = y_train / price_train - 1.0
            predictions = {"drift": price_val * (1.0 + float(np.mean(simple_train)))}
            for name in LEVEL_MODELS:
                predictions[name] = _fit_predict(
                    name, xp_train, y_train, xp_val, price_train, price_val
                )
            for name in RESIDUAL_MODELS + LOGRET_MODELS:
                predictions[name] = _fit_predict(
                    name, x13_train, y_train, x13_val, price_train, price_val
                )

            for name, pred in predictions.items():
                rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
                implied_return = pred / price_val - 1.0
                return_rmse = float(
                    np.sqrt(np.mean((actual_return - implied_return) ** 2))
                )
                rows.append(
                    {
                        "anchor": anchor,
                        "fold": fold.fold_id,
                        "val_year": int(fold.val_start.year),
                        "n": len(y_val),
                        "model": name,
                        "rmse": rmse,
                        "naive_rmse": naive_rmse,
                        "skill": 1.0 - rmse / naive_rmse,
                        "return_rmse": return_rmse,
                        "return_zero_rmse": return_zero_rmse,
                        "return_skill": 1.0 - return_rmse / return_zero_rmse,
                        "train_max": train_max,
                        "val_above_train_max": above,
                        "pred_mean": float(pred.mean()),
                        "val_mean": float(price_val.mean()),
                    }
                )

    detail = pd.DataFrame(rows)
    for anchor in ("log", "raw"):
        part = detail[detail["anchor"] == anchor]
        print(f"\n=== anchor: {anchor}(P_t) ===")
        print("Per-fold price-RMSE skill (positive = beats P_t):")
        print(
            part.pivot_table(index="fold", columns="model", values="skill")
            .round(4)
            .to_string()
        )
        print("\nPooled:")
        for name in ALL_MODELS:
            sub = part[part["model"] == name]
            n = sub["n"].sum()
            rmse = float(np.sqrt((sub["n"] * sub["rmse"] ** 2).sum() / n))
            naive = float(np.sqrt((sub["n"] * sub["naive_rmse"] ** 2).sum() / n))
            rrmse = float(np.sqrt((sub["n"] * sub["return_rmse"] ** 2).sum() / n))
            rzero = float(
                np.sqrt((sub["n"] * sub["return_zero_rmse"] ** 2).sum() / n)
            )
            print(
                f"  {name:16s} price skill {1 - rmse / naive:+8.2%} "
                f"(folds {int((sub['skill'] > 0).sum())}/{len(sub)})  "
                f"return skill {1 - rrmse / rzero:+8.2%}"
            )

    print("\nXGBoost extrapolation check (validation prices above training max):")
    print(
        detail[detail["model"] == "xgboost"][
            ["fold", "val_year", "train_max", "val_above_train_max"]
        ]
        .drop_duplicates("fold")
        .round(3)
        .to_string(index=False)
    )

    out = ROOT / "results" / "analysis" / "price_level_pilot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
