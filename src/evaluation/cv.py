"""Cross-validation runner: hyperparameter selection, final fit, backtest.

For each outer fold:

1. Fit the scaler on the selection-train window only.
2. Score every candidate on the validation year (RMSE).
3. Choose the signal threshold on the same validation predictions (Sharpe).
4. Refit the winning configuration on train + validation.
5. Predict the test year, backtest, and persist predictions and metrics.

Models may implement ``fit_with_validation`` for early stopping.  Online
models must implement ``partial_fit``; the runner then predicts one day,
records it, and only afterwards updates with the realised label.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COST_BPS,
    DATA_DIR,
    FEATURES,
    GLOBAL_SEED,
    HOLDOUT_SEED,
    METRICS_DIR,
    PREDS_DIR,
    PROTOCOL_VERSION,
    PROJECT_ROOT,
    RESULTS_DIR,
    TARGET_COL,
    TAU_GRID,
    fold_seed,
)
from src.evaluation.backtest import (
    backtest_metrics,
    make_signal,
    run_backtest,
    select_tau,
)
from src.evaluation.metrics import compute_metrics
from src.features import fit_scaler, transform_scaler
from src.splits import Fold


SUMMARY_COLUMNS = [
    "model",
    "scope",
    "rep",
    "fold",
    "n",
    "rmse",
    "mae",
    "r2",
    "ic",
    "da",
    "da5",
    "sharpe",
    "ann_ret",
    "max_dd",
    "turnover",
    "exposure",
    "tau",
]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_sequences(
    full_X: np.ndarray,
    idx: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal windows ending at each index in ``idx``.

    Returns the 3-D array and the positions that had enough history.  Windows
    may reach back into earlier rows of the same fold, which is allowed because
    those rows precede the prediction date.
    """
    idx = np.sort(np.asarray(idx, dtype=int))
    valid = idx[idx >= seq_len - 1]
    if valid.size == 0:
        raise ValueError("no rows have enough history for the requested sequence length")
    sequences = np.stack([full_X[i - seq_len + 1 : i + 1] for i in valid])
    return sequences, valid


def run_fold(
    model_name: str,
    model_factory,
    fold: Fold,
    frame: pd.DataFrame,
    param_grid: list[dict],
    *,
    seed: int,
    online: bool = False,
    cost_bps: float = COST_BPS,
    tau_grid=TAU_GRID,
    sequence_len: int | None = None,
    selection_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run one fold and return the prediction frame plus a metrics dict."""
    if sequence_len is not None and online:
        raise NotImplementedError("online sequence models are not supported")
    if selection_frame is not None and sequence_len is not None:
        raise NotImplementedError("separate selection market requires tabular inputs")

    selection_data = frame if selection_frame is None else selection_frame
    if len(selection_data) != len(frame) or not np.array_equal(
        selection_data["label_date"].to_numpy(), frame["label_date"].to_numpy()
    ):
        raise ValueError("selection and target frames must align on label_date")
    select_train = selection_data.iloc[fold.select_train_idx]
    val = selection_data.iloc[fold.val_idx]
    final_train = frame.iloc[fold.train_idx]
    test = frame.iloc[fold.test_idx]
    y_all = frame[TARGET_COL].to_numpy(dtype=np.float64)

    # --- hyperparameter selection on the selection-train + validation years ---
    scaler_select = fit_scaler(select_train[FEATURES])
    if sequence_len is None:
        X_select = transform_scaler(select_train[FEATURES], scaler_select)
        y_select = select_train[TARGET_COL].to_numpy(dtype=np.float64)
        X_val = transform_scaler(val[FEATURES], scaler_select)
        y_val = val[TARGET_COL].to_numpy(dtype=np.float64)
    else:
        full_select = transform_scaler(frame[FEATURES], scaler_select)
        X_select, select_valid = build_sequences(
            full_select, fold.select_train_idx, sequence_len
        )
        y_select = y_all[select_valid]
        X_val, val_valid = build_sequences(full_select, fold.val_idx, sequence_len)
        y_val = y_all[val_valid]

    best_model = None
    best_params: dict | None = None
    best_val_pred: np.ndarray | None = None
    best_score = np.inf
    best_tie = (np.inf, np.inf)
    search_results = []
    started = time.perf_counter()

    for params in param_grid:
        model = model_factory(params, seed)
        if hasattr(model, "fit_with_validation"):
            model.fit_with_validation(X_select, y_select, X_val, y_val)
        else:
            model.fit(X_select, y_select)
        pred_val = np.asarray(model.predict(X_val), dtype=np.float64)
        if not np.isfinite(pred_val).all() or pred_val.shape != y_val.shape:
            raise ValueError(f"{fold.fold_id}: invalid validation predictions")
        score = _mae_rmse(y_val, pred_val)
        tie = (getattr(model, "n_rules_", np.inf), -float(params.get("dthr", 0)))
        search_results.append({"params": dict(params), "val_rmse": score,
                               "n_rules": getattr(model, "n_rules_", None)})
        if score < best_score or (score == best_score and tie < best_tie):
            best_score = score
            best_tie = tie
            best_model = model
            best_params = dict(params)
            best_val_pred = pred_val

    if best_model is None or best_params is None or best_val_pred is None:
        raise RuntimeError(f"{fold.fold_id}: empty parameter grid")

    tau = select_tau(y_val, best_val_pred, tau_grid=tau_grid, cost_bps=cost_bps)

    final_params = dict(best_params)
    best_iteration = getattr(best_model, "best_iteration_", None)
    if best_iteration is not None:
        final_params["n_estimators"] = int(best_iteration) + 1
    best_epoch = getattr(best_model, "best_epoch_", None)
    if best_epoch is not None:
        final_params["epochs"] = int(best_epoch)

    # --- final fit on train + validation, scaler refit on that window ---
    scaler_final = fit_scaler(final_train[FEATURES])
    if sequence_len is None:
        X_final = transform_scaler(final_train[FEATURES], scaler_final)
        y_final = final_train[TARGET_COL].to_numpy(dtype=np.float64)
        X_test = transform_scaler(test[FEATURES], scaler_final)
        y_test = test[TARGET_COL].to_numpy(dtype=np.float64)
        test_meta = test
    else:
        full_final = transform_scaler(frame[FEATURES], scaler_final)
        X_final, final_valid = build_sequences(
            full_final, fold.train_idx, sequence_len
        )
        y_final = y_all[final_valid]
        X_test, test_valid = build_sequences(
            full_final, fold.test_idx, sequence_len
        )
        y_test = y_all[test_valid]
        test_meta = frame.iloc[test_valid]

    final_model = model_factory(final_params, seed)
    final_model.fit(X_final, y_final)

    if online:
        if not hasattr(final_model, "partial_fit"):
            raise TypeError(f"{model_name} is marked online but has no partial_fit")
        pred_test = np.empty(len(y_test), dtype=np.float64)
        for i in range(len(y_test)):
            pred_test[i] = float(
                np.asarray(final_model.predict(X_test[i : i + 1]), dtype=np.float64)[0]
            )
            final_model.partial_fit(X_test[i], y_test[i])
    else:
        pred_test = np.asarray(final_model.predict(X_test), dtype=np.float64)

    if pred_test.shape != y_test.shape or not np.isfinite(pred_test).all():
        raise ValueError(f"{fold.fold_id}: invalid test predictions")

    # --- metrics and backtest ---
    metrics = compute_metrics(y_test, pred_test)
    signal = make_signal(pred_test, tau)
    simple_returns = np.expm1(y_test)
    backtest = run_backtest(signal, simple_returns, cost_bps=cost_bps)
    metrics.update(backtest_metrics(backtest["strat_ret"], backtest["signal"]))
    metrics.update(
        {
            "model": model_name,
            "fold_id": fold.fold_id,
            "rep": fold.rep,
            "fold": fold.fold,
            "test_year": fold.test_year,
            "tau": float(tau),
            "seed": int(seed),
            "params": final_params,
            "validation_rmse": float(best_score),
            "search_results": search_results,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if hasattr(final_model, "n_rules_"):
        metrics.update({
            "n_rules": int(final_model.n_rules_),
            "rule_history": list(getattr(final_model, "rule_history_", [])),
            "covariance_resets": int(getattr(final_model, "resets_", 0)),
            "cap_hits": int(getattr(final_model, "cap_hits_", 0)),
            "merges": int(getattr(final_model, "merges_", 0)),
        })

    pred_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(test_meta["label_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "y_true": y_test,
            "y_pred": pred_test,
            "signal": backtest["signal"].to_numpy(),
            "position": backtest["position"].to_numpy(),
            "strat_ret": backtest["strat_ret"].to_numpy(),
            "bench_ret": backtest["bench_ret"].to_numpy(),
        }
    )
    return pred_frame, metrics


def save_fold_result(
    model_name: str,
    fold_id: str,
    pred_frame: pd.DataFrame,
    metrics: dict,
) -> tuple[Path, Path]:
    pred_path = PREDS_DIR / model_name / f"{fold_id}.csv"
    metric_path = METRICS_DIR / model_name / f"{fold_id}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    pred_frame.to_csv(pred_path, index=False, float_format="%.10g")
    metric_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return pred_path, metric_path


def run_model_on_folds(
    model_name: str,
    model_factory,
    folds: list[Fold],
    frame: pd.DataFrame,
    param_grid: list[dict],
    *,
    online: bool = False,
    fold_ids: list[str] | None = None,
    cost_bps: float = COST_BPS,
    sequence_len: int | None = None,
) -> pd.DataFrame:
    """Run a model over selected folds and refresh summary + manifest."""
    selected = [
        fold for fold in folds
        if (fold.fold_id != "holdout" if fold_ids is None
            else fold.fold_id in set(fold_ids))
    ]
    if fold_ids is not None:
        unknown = set(fold_ids) - {fold.fold_id for fold in folds}
        if unknown:
            raise ValueError(f"unknown fold ids: {sorted(unknown)}")
    rows = []
    for fold in selected:
        seed = HOLDOUT_SEED if fold.fold_id == "holdout" else fold_seed(fold.rep, fold.fold)
        pred_frame, metrics = run_fold(
            model_name,
            model_factory,
            fold,
            frame,
            param_grid,
            seed=seed,
            online=online,
            cost_bps=cost_bps,
            sequence_len=sequence_len,
        )
        save_fold_result(model_name, fold.fold_id, pred_frame, metrics)
        rows.append(metrics)
        print(
            f"{model_name:14s} {fold.fold_id:7s} "
            f"rmse={metrics['rmse']:.6f}  da={metrics['da']:.3f}  "
            f"sharpe={metrics['sharpe']:.3f}  tau={metrics['tau']:.4f}"
        )

    summary = build_summary()
    manifest = write_manifest()
    if isinstance(manifest, dict):
        manifest = {**manifest, "model": model_name,
                    "folds": [fold.fold_id for fold in selected]}
        directory = RESULTS_DIR / "manifests"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{model_name}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def build_summary() -> pd.DataFrame:
    """Rebuild results/summary.csv from every metrics JSON on disk."""
    rows: list[dict] = []
    metric_files = sorted(METRICS_DIR.glob("*/*.json"))
    if not metric_files:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    by_model: dict[str, list[tuple[str, dict]]] = {}
    for path in metric_files:
        model = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8"))
        fold_id = str(data["fold_id"])
        scope = "holdout" if fold_id == "holdout" else "fold"
        row = {
            "model": model,
            "scope": scope,
            "rep": data.get("rep"),
            "fold": data.get("fold"),
            "n": data.get("n"),
            "rmse": data.get("rmse"),
            "mae": data.get("mae"),
            "r2": data.get("r2"),
            "ic": data.get("ic"),
            "da": data.get("da"),
            "da5": data.get("da5"),
            "sharpe": data.get("sharpe"),
            "ann_ret": data.get("ann_ret"),
            "max_dd": data.get("max_dd"),
            "turnover": data.get("turnover"),
            "exposure": data.get("exposure"),
            "tau": data.get("tau"),
        }
        rows.append(row)
        by_model.setdefault(model, []).append((fold_id, data))

    # Pooled metrics across the nine regular folds (holdout excluded).
    for model, items in by_model.items():
        regular = [(fid, data) for fid, data in items if fid != "holdout"]
        if not regular:
            continue
        preds = []
        for fold_id, _ in regular:
            path = PREDS_DIR / model / f"{fold_id}.csv"
            if path.exists():
                preds.append(pd.read_csv(path))
        if not preds:
            continue

        pooled = pd.concat(preds, ignore_index=True).sort_values("date")
        pooled_metrics = compute_metrics(pooled["y_true"], pooled["y_pred"])
        pooled_backtest = backtest_metrics(pooled["strat_ret"], pooled["signal"])

        # Recompute turnover/exposure as row-weighted fold averages so that
        # fold-boundary position resets are respected.
        weights = np.array([data["n"] for _, data in regular], dtype=float)
        pooled_backtest["turnover"] = float(
            np.average([data["turnover"] for _, data in regular], weights=weights)
        )
        pooled_backtest["exposure"] = float(
            np.average([data["exposure"] for _, data in regular], weights=weights)
        )

        rows.append(
            {
                "model": model,
                "scope": "pooled",
                "rep": None,
                "fold": None,
                "n": pooled_metrics["n"],
                "rmse": pooled_metrics["rmse"],
                "mae": pooled_metrics["mae"],
                "r2": pooled_metrics["r2"],
                "ic": pooled_metrics["ic"],
                "da": pooled_metrics["da"],
                "da5": pooled_metrics["da5"],
                "sharpe": pooled_backtest["sharpe"],
                "ann_ret": pooled_backtest["ann_ret"],
                "max_dd": pooled_backtest["max_dd"],
                "turnover": pooled_backtest["turnover"],
                "exposure": pooled_backtest["exposure"],
                "tau": None,
            }
        )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(
        ["model", "scope", "rep", "fold"], na_position="last"
    ).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "torch",
        "matplotlib",
        "scipy",
        "yfinance",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def write_manifest() -> dict:
    data_sha: dict[str, str] = {}
    checksum_file = DATA_DIR / "checksums.txt"
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                path = DATA_DIR / parts[1]
                data_sha[parts[1]] = _sha256(path)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "config_hash": _sha256(PROJECT_ROOT / "config.py"),
        "git_commit": _git_commit(),
        "python": sys.version,
        "executable": sys.executable,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for root in ("src", "scripts")
            for path in sorted((PROJECT_ROOT / root).rglob("*.py"))
        },
        "packages": _package_versions(),
        "data_sha256": data_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
