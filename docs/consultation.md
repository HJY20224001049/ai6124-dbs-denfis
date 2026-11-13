# Project consultation — preparation

Status: in-person consultation on the evening of 2026-11-11. LIN Jingzhao leads;
HU Junyu records. Update the answers below during or immediately after the meeting.

## Already answered by email (2026-11-11)

1. "Better than the original baselines" is an ordering on forecast error: DENFIS
   below XGBoost and LSTM, and both baselines below no prediction. The metric is
   our choice but must include the no-prediction comparison.
2. Under that definition the current outcome does **not** qualify.
3. Whether a documented negative result is acceptable: not answered; the
   instructor suspects an implementation error instead.
4. Which route to take if a positive result is required: not answered; he offered
   to discuss after the lecture.

## Tonight's agenda

Bring the updated `docs/consultation_handout.md` / `writing/consultation_handout.pdf`,
`docs/price_level_pilot.md`, `scripts/synthetic_signal_check.py`,
`docs/results_table.md`, `docs/statistical_tests.md` and the final notebook.

Three checks to present:

- Target is `ln(P_{t+1}/P_t)` (price difference), not the price level
  (`src/features.py:87`); the price-level pilot shows a random walk is almost
  unbeatable and XGBoost cannot extrapolate.
- On a synthetic target with a known signal, the same pipeline beats zero by
  2.45% (DENFIS), 2.10% (XGBoost) and 1.48% (LSTM), so the implementation learns.
- All four models lose to no prediction on the real data (0.28%–1.76% worse
  RMSE), so the shortfall is not specific to DENFIS.

Five questions to settle:

1. Price level or price difference (log return) as the target?
2. What further check would rule out an implementation error?
3. Does the error ordering need statistical significance (DM p < 0.05), or is a
   point-estimate ordering sufficient?
4. If even the baselines lose to no prediction, is the daily DBS task solvable as
   specified, or is our data/target setup still suspect?
5. If a negative result is unacceptable, which route is authorised: (a) change
   the target, (b) report the ensemble as a separate system result, or (c) keep
   the protocol and report the negative result? Written confirmation is needed
   either way.

## Answers

Pending. Protocol changes: none authorised by the instructor yet.
