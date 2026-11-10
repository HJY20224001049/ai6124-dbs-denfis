"""Read-only acceptance audit for method-line predictions and transfer cutoffs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import COST_BPS, FEATURES, TARGET_COL
from src.data import load_prices
from src.features import compute_features
from src.features import fit_scaler, transform_scaler
from src.models.denfis import DENFIS
from src.splits import make_folds_from_frame
from src.transfer import load_transfer_frames
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import compute_metrics


def check_predictions(path, metrics, frame, fold):
    saved = pd.read_csv(path)
    expected = frame.iloc[fold.test_idx]
    assert saved.date.tolist() == expected.label_date.dt.strftime('%Y-%m-%d').tolist(), path
    assert np.isfinite(saved.select_dtypes(include='number').to_numpy()).all(), path
    np.testing.assert_allclose(saved.y_true, expected[TARGET_COL], atol=1e-10, rtol=0)
    recomputed = compute_metrics(saved.y_true, saved.y_pred)
    assert abs(recomputed['rmse']-metrics['rmse']) < 1e-9, path
    expected_signals = (saved.y_pred.to_numpy() > metrics['tau']).astype(int)
    np.testing.assert_array_equal(saved.signal, expected_signals)
    ledger = run_backtest(saved.signal.to_numpy(), np.expm1(saved.y_true.to_numpy()), cost_bps=COST_BPS)
    np.testing.assert_allclose(saved.strat_ret, ledger.strat_ret, atol=1e-9, rtol=0)
    assert 1 <= metrics['n_rules'] <= 200, path
    if 'online' in str(path):
        assert len(metrics['rule_history']) == len(saved)+1
    return len(saved)


def main():
    frame = compute_features(load_prices('D05_SI'))
    folds = {f.fold_id:f for f in make_folds_from_frame(frame) if f.fold_id != 'holdout'}
    count, rows = 0, 0
    for model in ('denfis_frozen','denfis_online'):
        directory = ROOT/'results/preds'/model
        assert {p.stem for p in directory.glob('*.csv')} == set(folds)
        for fid, fold in folds.items():
            metrics = json.loads((ROOT/'results/metrics'/model/f'{fid}.json').read_text())
            assert len(metrics['search_results']) == 20
            assert metrics['validation_rmse'] == min(x['val_rmse'] for x in metrics['search_results'])
            rows += check_predictions(directory/f'{fid}.csv', metrics, frame, fold)
            # Replay only the saved winner, not another hyperparameter search.
            train, test = frame.iloc[fold.train_idx], frame.iloc[fold.test_idx]
            scaler = fit_scaler(train[FEATURES])
            X_train = transform_scaler(train[FEATURES], scaler)
            X_test = transform_scaler(test[FEATURES], scaler)
            replay = DENFIS(**metrics['params'], seed=metrics['seed'],
                            evolve_rules=model == 'denfis_online')
            replay.fit(X_train, train[TARGET_COL].to_numpy())
            if model == 'denfis_online':
                predictions = []
                for x, y in zip(X_test, test[TARGET_COL]):
                    predictions.append(replay.predict(x[None])[0])
                    replay.partial_fit(x, float(y))
            else:
                predictions = replay.predict(X_test)
            saved = pd.read_csv(directory/f'{fid}.csv')
            np.testing.assert_allclose(predictions, saved.y_pred, atol=1e-10, rtol=0)
            count += 1
    transfer = load_transfer_frames()['DBS']
    transfer_folds = {f.fold_id:f for f in make_folds_from_frame(transfer)}
    for protocol in ('scratch','zero_shot','fine_tune'):
        for fid in ('r1f1','r2f2','r3f3'):
            metric_path = ROOT/'results/transfer/metrics'/protocol/f'{fid}.json'
            metrics = json.loads(metric_path.read_text())
            fold = transfer_folds[fid]
            assert pd.Timestamp(metrics['source_select_end']) <= fold.select_train_end
            assert pd.Timestamp(metrics['source_final_end']) <= fold.final_train_end
            assert pd.Timestamp(metrics['source_final_end']) < pd.Timestamp(metrics['test_start'])
            assert metrics['test_adaptation'] is False
            if protocol == 'zero_shot':
                assert metrics['selection_market'] == 'STI'
            rows += check_predictions(ROOT/'results/transfer/preds'/protocol/f'{fid}.csv', metrics, transfer, fold)
            count += 1
    for variant in ('no_rls', 'no_merge'):
        for fid in ('r1f1', 'r2f1', 'r2f2', 'r3f3'):
            metrics = json.loads((ROOT/'results/analysis/metrics'/variant/f'{fid}.json').read_text())
            rows += check_predictions(ROOT/'results/analysis/preds'/variant/f'{fid}.csv', metrics, frame, folds[fid])
            count += 1
    report = {'prediction_files':count,'prediction_rows_checked':rows,
              'regular_folds_per_main_model':9,'transfer_protocols':3,
              'transfer_folds_per_protocol':3,'denfis_holdout_run':False,
              'selected_model_replays':18,
              'checks':'date/target alignment; finite values; RMSE; signal threshold; cost ledger; rule cap; selection records; source cutoffs; online history',
              'status':'PASS'}
    (ROOT/'docs/method_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
