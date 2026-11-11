# Leakage audit

Checks: 151; failures: 0

| Check | Result | Detail |
|---|---|---|
| dates are sorted and unique | PASS |  |
| every label date follows its feature date | PASS |  |
| label_date is the next available trading row | PASS |  |
| no NaN in features or target | PASS |  |
| r1f1: train ends before test | PASS | 2016-12-30 < 2017-01-03 |
| r1f1: validation ends before test | PASS | 2016-12-30 < 2017-01-03 |
| r1f1: selection train ends before validation | PASS | 2015-12-31 < 2016-01-01 |
| r1f1: test inside its window | PASS |  |
| r1f1: no train/test index overlap | PASS |  |
| r1f2: train ends before test | PASS | 2017-12-29 < 2018-01-02 |
| r1f2: validation ends before test | PASS | 2017-12-29 < 2018-01-02 |
| r1f2: selection train ends before validation | PASS | 2016-12-30 < 2017-01-01 |
| r1f2: test inside its window | PASS |  |
| r1f2: no train/test index overlap | PASS |  |
| r1f3: train ends before test | PASS | 2018-12-31 < 2019-01-02 |
| r1f3: validation ends before test | PASS | 2018-12-31 < 2019-01-02 |
| r1f3: selection train ends before validation | PASS | 2017-12-29 < 2018-01-01 |
| r1f3: test inside its window | PASS |  |
| r1f3: no train/test index overlap | PASS |  |
| r2f1: train ends before test | PASS | 2019-12-31 < 2020-01-02 |
| r2f1: validation ends before test | PASS | 2019-12-31 < 2020-01-02 |
| r2f1: selection train ends before validation | PASS | 2018-12-31 < 2019-01-01 |
| r2f1: test inside its window | PASS |  |
| r2f1: no train/test index overlap | PASS |  |
| r2f2: train ends before test | PASS | 2020-12-31 < 2021-01-04 |
| r2f2: validation ends before test | PASS | 2020-12-31 < 2021-01-04 |
| r2f2: selection train ends before validation | PASS | 2019-12-31 < 2020-01-01 |
| r2f2: test inside its window | PASS |  |
| r2f2: no train/test index overlap | PASS |  |
| r2f3: train ends before test | PASS | 2021-12-31 < 2022-01-03 |
| r2f3: validation ends before test | PASS | 2021-12-31 < 2022-01-03 |
| r2f3: selection train ends before validation | PASS | 2020-12-31 < 2021-01-01 |
| r2f3: test inside its window | PASS |  |
| r2f3: no train/test index overlap | PASS |  |
| r3f1: train ends before test | PASS | 2022-12-30 < 2023-01-03 |
| r3f1: validation ends before test | PASS | 2022-12-30 < 2023-01-03 |
| r3f1: selection train ends before validation | PASS | 2021-12-31 < 2022-01-01 |
| r3f1: test inside its window | PASS |  |
| r3f1: no train/test index overlap | PASS |  |
| r3f2: train ends before test | PASS | 2023-12-29 < 2024-01-02 |
| r3f2: validation ends before test | PASS | 2023-12-29 < 2024-01-02 |
| r3f2: selection train ends before validation | PASS | 2022-12-30 < 2023-01-01 |
| r3f2: test inside its window | PASS |  |
| r3f2: no train/test index overlap | PASS |  |
| r3f3: train ends before test | PASS | 2024-12-31 < 2025-01-02 |
| r3f3: validation ends before test | PASS | 2024-12-31 < 2025-01-02 |
| r3f3: selection train ends before validation | PASS | 2023-12-29 < 2024-01-01 |
| r3f3: test inside its window | PASS |  |
| r3f3: no train/test index overlap | PASS |  |
| holdout: train ends before test | PASS | 2025-12-31 < 2026-01-02 |
| holdout: validation ends before test | PASS | 2025-12-31 < 2026-01-02 |
| holdout: selection train ends before validation | PASS | 2024-12-31 < 2025-01-01 |
| holdout: test inside its window | PASS |  |
| holdout: no train/test index overlap | PASS |  |
| features before 2020-01-02 unchanged when future prices double | PASS | 2444 rows compared |
| scaler parameters unchanged when validation/test features are perturbed | PASS | fold r1f1 |
| denfis_frozen/r1f1: dates and y align | PASS | 250 rows |
| denfis_frozen/r1f1: finite predictions | PASS |  |
| denfis_frozen/r1f2: dates and y align | PASS | 251 rows |
| denfis_frozen/r1f2: finite predictions | PASS |  |
| denfis_frozen/r1f3: dates and y align | PASS | 250 rows |
| denfis_frozen/r1f3: finite predictions | PASS |  |
| denfis_frozen/r2f1: dates and y align | PASS | 251 rows |
| denfis_frozen/r2f1: finite predictions | PASS |  |
| denfis_frozen/r2f2: dates and y align | PASS | 253 rows |
| denfis_frozen/r2f2: finite predictions | PASS |  |
| denfis_frozen/r2f3: dates and y align | PASS | 250 rows |
| denfis_frozen/r2f3: finite predictions | PASS |  |
| denfis_frozen/r3f1: dates and y align | PASS | 249 rows |
| denfis_frozen/r3f1: finite predictions | PASS |  |
| denfis_frozen/r3f2: dates and y align | PASS | 252 rows |
| denfis_frozen/r3f2: finite predictions | PASS |  |
| denfis_frozen/r3f3: dates and y align | PASS | 252 rows |
| denfis_frozen/r3f3: finite predictions | PASS |  |
| denfis_online/r1f1: dates and y align | PASS | 250 rows |
| denfis_online/r1f1: finite predictions | PASS |  |
| denfis_online/r1f2: dates and y align | PASS | 251 rows |
| denfis_online/r1f2: finite predictions | PASS |  |
| denfis_online/r1f3: dates and y align | PASS | 250 rows |
| denfis_online/r1f3: finite predictions | PASS |  |
| denfis_online/r2f1: dates and y align | PASS | 251 rows |
| denfis_online/r2f1: finite predictions | PASS |  |
| denfis_online/r2f2: dates and y align | PASS | 253 rows |
| denfis_online/r2f2: finite predictions | PASS |  |
| denfis_online/r2f3: dates and y align | PASS | 250 rows |
| denfis_online/r2f3: finite predictions | PASS |  |
| denfis_online/r3f1: dates and y align | PASS | 249 rows |
| denfis_online/r3f1: finite predictions | PASS |  |
| denfis_online/r3f2: dates and y align | PASS | 252 rows |
| denfis_online/r3f2: finite predictions | PASS |  |
| denfis_online/r3f3: dates and y align | PASS | 252 rows |
| denfis_online/r3f3: finite predictions | PASS |  |
| lstm/holdout: dates and y align | PASS | 209 rows |
| lstm/holdout: finite predictions | PASS |  |
| lstm/r1f1: dates and y align | PASS | 250 rows |
| lstm/r1f1: finite predictions | PASS |  |
| lstm/r1f2: dates and y align | PASS | 251 rows |
| lstm/r1f2: finite predictions | PASS |  |
| lstm/r1f3: dates and y align | PASS | 250 rows |
| lstm/r1f3: finite predictions | PASS |  |
| lstm/r2f1: dates and y align | PASS | 251 rows |
| lstm/r2f1: finite predictions | PASS |  |
| lstm/r2f2: dates and y align | PASS | 253 rows |
| lstm/r2f2: finite predictions | PASS |  |
| lstm/r2f3: dates and y align | PASS | 250 rows |
| lstm/r2f3: finite predictions | PASS |  |
| lstm/r3f1: dates and y align | PASS | 249 rows |
| lstm/r3f1: finite predictions | PASS |  |
| lstm/r3f2: dates and y align | PASS | 252 rows |
| lstm/r3f2: finite predictions | PASS |  |
| lstm/r3f3: dates and y align | PASS | 252 rows |
| lstm/r3f3: finite predictions | PASS |  |
| no_prediction/r1f1: dates and y align | PASS | 250 rows |
| no_prediction/r1f1: finite predictions | PASS |  |
| no_prediction/r1f2: dates and y align | PASS | 251 rows |
| no_prediction/r1f2: finite predictions | PASS |  |
| no_prediction/r1f3: dates and y align | PASS | 250 rows |
| no_prediction/r1f3: finite predictions | PASS |  |
| no_prediction/r2f1: dates and y align | PASS | 251 rows |
| no_prediction/r2f1: finite predictions | PASS |  |
| no_prediction/r2f2: dates and y align | PASS | 253 rows |
| no_prediction/r2f2: finite predictions | PASS |  |
| no_prediction/r2f3: dates and y align | PASS | 250 rows |
| no_prediction/r2f3: finite predictions | PASS |  |
| no_prediction/r3f1: dates and y align | PASS | 249 rows |
| no_prediction/r3f1: finite predictions | PASS |  |
| no_prediction/r3f2: dates and y align | PASS | 252 rows |
| no_prediction/r3f2: finite predictions | PASS |  |
| no_prediction/r3f3: dates and y align | PASS | 252 rows |
| no_prediction/r3f3: finite predictions | PASS |  |
| xgboost/holdout: dates and y align | PASS | 209 rows |
| xgboost/holdout: finite predictions | PASS |  |
| xgboost/r1f1: dates and y align | PASS | 250 rows |
| xgboost/r1f1: finite predictions | PASS |  |
| xgboost/r1f2: dates and y align | PASS | 251 rows |
| xgboost/r1f2: finite predictions | PASS |  |
| xgboost/r1f3: dates and y align | PASS | 250 rows |
| xgboost/r1f3: finite predictions | PASS |  |
| xgboost/r2f1: dates and y align | PASS | 251 rows |
| xgboost/r2f1: finite predictions | PASS |  |
| xgboost/r2f2: dates and y align | PASS | 253 rows |
| xgboost/r2f2: finite predictions | PASS |  |
| xgboost/r2f3: dates and y align | PASS | 250 rows |
| xgboost/r2f3: finite predictions | PASS |  |
| xgboost/r3f1: dates and y align | PASS | 249 rows |
| xgboost/r3f1: finite predictions | PASS |  |
| xgboost/r3f2: dates and y align | PASS | 252 rows |
| xgboost/r3f2: finite predictions | PASS |  |
| xgboost/r3f3: dates and y align | PASS | 252 rows |
| xgboost/r3f3: finite predictions | PASS |  |
| at least one prediction file audited | PASS | 47 files |
