# Method presentation rehearsal script (draft)

Target: five slides, approximately 4:35–4:55. Actual timed rehearsal is pending.
The complete two-person talk must stay within ten minutes.

1. **Problem (45 s).** We predict next-day DBS returns from thirteen causal
   features. Our question is whether evolving local fuzzy rules provide useful
   forecasts and an interpretable account of their output. Nine chronological
   years form the main evaluation; the final holdout is reserved.
2. **Architecture (65 s).** Explain the ECM centre/radius picture, Gaussian
   activation and local linear TSK outputs. Weighted RLS estimates consequents.
   State the two-pass training and bounded-rule deviations rather than claiming
   exact reproduction of every original DENFIS option.
3. **Evaluation (55 s).** Parameters come from validation RMSE. Frozen evaluation
   does not update. Online evaluation predicts first and updates only after the
   label is realised; it belongs in a separate extension comparison.
4. **Evidence (65 s).** Read the current pooled values from the generated slide.
   Separate forecast error from strategy Sharpe. Compare uncertainty and exposure
   before claiming any improvement; buy-and-hold remains an essential reference.
5. **Explanation and limits (55 s).** Show rule contributions for a difficult
   high-volatility day. They explain the numerical prediction, not economic
   causality. Briefly mention transfer, ablations, single-asset limits and next
   steps. Hand over to HU Junyu for the broader evaluation discussion.

Do not state a rehearsal duration as measured until the students actually rehearse.
