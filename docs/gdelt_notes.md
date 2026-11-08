# GDELT sentiment — attempt log

**Date:** 2026-11-08
**Owner:** B (data & evaluation)
**Status:** attempted, not available from the current environment; dropped from main results.

## What was tried

`GDELT DOC 2.0` API, average-tone timeline:

```text
GET https://api.gdeltproject.org/api/v2/doc/doc
    ?query="DBS"
    &mode=timelinetone
    &format=json
    &startdatetime=20170101000000
    &enddatetime=20260101000000
    &timelinesmooth=0
```

Two requests were made, seven seconds apart.

## Result

Both returned **HTTP 429** with:

> Please limit requests to one every 5 seconds or contact ... for larger queries.
> All high-traffic users should switch to our ngrams dataset ...

The endpoint stayed rate-limited after the pause, so the timebox was stopped.

## Decision

- Do **not** include GDELT in the main results or the primary comparison.
- Report it in the bonus/limitations section as an attempted external data
  source that was unavailable, with the exact endpoint and error.
- The transfer-learning bonus (^STI pre-training) is unaffected and remains the
  primary bonus experiment.

## If the endpoint becomes available later

1. Pull daily average tone for `"DBS" OR "Singapore banks"`.
2. Build causal features: `tone`, 5-day mean, 20-day mean, 1-day change.
3. Merge by date; missing days carry the previous value or 0, documented.
4. Because the DOC API starts in 2017, restrict the ablation to folds whose
   training and test windows are both after 2017 (e.g. `r2f3`, `r3f2`, `r3f3`).
5. Keep the GDELT run as a separate table, never pooled with the main models.
