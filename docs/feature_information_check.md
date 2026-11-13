# Do the 13 features carry next-day information?

Diagnostic run by `scripts/feature_information_check.py`. Nothing here selects
a feature, hyper-parameter or fold on test performance, and the 2026 holdout
is untouched. The test table evaluates rules fitted on the final training
window; it is descriptive, not a new model selection.

## Validation years (fit on labels through Y-2, score Y-1)

| rule | pooled skill | fold-bootstrap 95% CI | folds > 0 | sign p | leave-one-out range | IC |
|---|---:|---|---:|---:|---|---:|
| ma5 | +0.317% | [-0.068%, +0.609%] | 5/9 | 1.000 | [+0.191%, +0.428%] | +0.0618 |
| boll_b | +0.264% | [-0.054%, +0.503%] | 6/9 | 0.508 | [+0.174%, +0.343%] | +0.0515 |
| ret_0 | +0.242% | [-0.118%, +0.568%] | 5/9 | 1.000 | [+0.148%, +0.347%] | +0.0484 |
| ret_1 | +0.116% | [-0.019%, +0.309%] | 6/9 | 0.508 | [+0.054%, +0.157%] | +0.0083 |
| ret_2 | +0.097% | [-0.017%, +0.252%] | 6/9 | 0.508 | [+0.057%, +0.123%] | -0.0121 |
| ret_3 | +0.113% | [+0.025%, +0.250%] | 7/9 | 0.180 | [+0.071%, +0.151%] | -0.0040 |
| ret_4 | +0.171% | [+0.042%, +0.342%] | 6/9 | 0.508 | [+0.119%, +0.208%] | +0.0303 |
| ma20 | +0.209% | [-0.041%, +0.403%] | 6/9 | 0.508 | [+0.138%, +0.270%] | +0.0390 |
| ma60 | +0.081% | [-0.029%, +0.246%] | 5/9 | 1.000 | [+0.030%, +0.117%] | -0.0298 |
| rsi14 | +0.164% | [-0.002%, +0.315%] | 6/9 | 0.508 | [+0.109%, +0.201%] | +0.0261 |
| macd | +0.211% | [-0.053%, +0.449%] | 5/9 | 1.000 | [+0.124%, +0.286%] | +0.0434 |
| rv20 | +0.090% | [-0.023%, +0.248%] | 5/9 | 1.000 | [+0.046%, +0.131%] | -0.0122 |
| vol_z | +0.096% | [-0.065%, +0.314%] | 7/9 | 0.180 | [+0.050%, +0.208%] | -0.0074 |
| CONST_MEAN | +0.113% | [+0.019%, +0.262%] | 7/9 | 0.180 | [+0.073%, +0.157%] | -0.0150 |
| ALL_LINEAR | -0.151% | [-0.527%, +0.331%] | 3/9 | 0.508 | [-0.277%, -0.039%] | +0.0259 |
| RIDGE | -0.130% | [-0.508%, +0.326%] | 3/9 | 0.508 | [-0.255%, -0.016%] | +0.0269 |

## Test years (fit on labels through Y-1, score Y; evaluation only)

| rule | pooled skill | fold-bootstrap 95% CI | folds > 0 | sign p | leave-one-out range | IC |
|---|---:|---|---:|---:|---|---:|
| ma5 | +0.393% | [-0.025%, +0.724%] | 5/9 | 1.000 | [+0.288%, +0.512%] | +0.0683 |
| boll_b | +0.240% | [-0.046%, +0.468%] | 6/9 | 0.508 | [+0.142%, +0.316%] | +0.0394 |
| ret_0 | +0.270% | [-0.099%, +0.619%] | 5/9 | 1.000 | [+0.179%, +0.377%] | +0.0470 |
| ret_1 | +0.161% | [+0.009%, +0.364%] | 6/9 | 0.508 | [+0.105%, +0.217%] | +0.0138 |
| ret_2 | +0.143% | [+0.012%, +0.305%] | 6/9 | 0.508 | [+0.107%, +0.183%] | -0.0046 |
| ret_3 | +0.145% | [+0.051%, +0.279%] | 8/9 | 0.039 | [+0.106%, +0.194%] | -0.0049 |
| ret_4 | +0.170% | [+0.042%, +0.347%] | 6/9 | 0.508 | [+0.117%, +0.205%] | +0.0188 |
| ma20 | +0.162% | [-0.062%, +0.333%] | 6/9 | 0.508 | [+0.076%, +0.217%] | +0.0197 |
| ma60 | +0.130% | [+0.020%, +0.290%] | 6/9 | 0.508 | [+0.085%, +0.184%] | -0.0265 |
| rsi14 | +0.186% | [+0.009%, +0.344%] | 6/9 | 0.508 | [+0.134%, +0.225%] | +0.0206 |
| macd | +0.112% | [-0.092%, +0.286%] | 5/9 | 1.000 | [+0.012%, +0.172%] | +0.0187 |
| rv20 | +0.145% | [+0.022%, +0.312%] | 6/9 | 0.508 | [+0.108%, +0.205%] | +0.0013 |
| vol_z | +0.092% | [-0.072%, +0.312%] | 7/9 | 0.180 | [+0.045%, +0.202%] | -0.0283 |
| CONST_MEAN | +0.148% | [+0.038%, +0.298%] | 7/9 | 0.180 | [+0.111%, +0.203%] | -0.0162 |
| ALL_LINEAR | -0.134% | [-0.511%, +0.355%] | 3/9 | 0.508 | [-0.258%, -0.021%] | +0.0166 |
| RIDGE | -0.121% | [-0.483%, +0.370%] | 3/9 | 0.508 | [-0.244%, -0.007%] | +0.0177 |
| no_prediction | +0.000% | [+0.000%, +0.000%] | 0/9 | 0.004 | [+0.000%, +0.000%] | +0.0000 |
| denfis_frozen | -0.759% | [-2.352%, +0.433%] | 2/9 | 0.180 | [-1.107%, -0.153%] | +0.0357 |
| denfis_online | -1.756% | [-3.273%, +0.048%] | 2/9 | 0.180 | [-2.176%, -1.043%] | +0.0012 |
| xgboost | -0.283% | [-1.016%, +0.319%] | 5/9 | 1.000 | [-0.425%, +0.019%] | -0.0094 |
| lstm | -0.569% | [-1.060%, +0.099%] | 2/9 | 0.180 | [-0.709%, -0.328%] | +0.0061 |

`no_prediction` is the benchmark row; its sign test is degenerate (all zeros)
and should be ignored.

## Simple combinations (features ranked on validation, evaluated on test)

| k | features | pooled skill | fold-bootstrap 95% CI | folds > 0 | sign p | IC |
|---:|---|---:|---|---:|---:|---:|
| 1 | ma5 | +0.393% | [-0.028%, +0.733%] | 5/9 | 1.000 | +0.0683 |
| 2 | ma5, boll_b | +0.346% | [-0.099%, +0.676%] | 5/9 | 1.000 | +0.0588 |
| 3 | ma5, boll_b, ret_0 | +0.264% | [-0.170%, +0.603%] | 6/9 | 0.508 | +0.0471 |
| 4 | ma5, boll_b, ret_0, macd | +0.178% | [-0.226%, +0.518%] | 5/9 | 1.000 | +0.0383 |
| 5 | ma5, boll_b, ret_0, macd, ma20 | +0.194% | [-0.208%, +0.586%] | 5/9 | 1.000 | +0.0445 |
| 6 | ma5, boll_b, ret_0, macd, ma20, ret_4 | +0.230% | [-0.181%, +0.622%] | 5/9 | 1.000 | +0.0498 |
| 7 | ma5, boll_b, ret_0, macd, ma20, ret_4, rsi14 | +0.196% | [-0.226%, +0.594%] | 5/9 | 1.000 | +0.0461 |
| 8 | ma5, boll_b, ret_0, macd, ma20, ret_4, rsi14, ret_1 | +0.156% | [-0.279%, +0.553%] | 5/9 | 1.000 | +0.0416 |

## Reading

- **There is weak univariate signal, but it is not fold-robust.** Several
  single-feature rules have fold-bootstrap intervals that exclude zero on the
  test years (`ret_1` to `ret_4`, `rsi14`, `rv20`, `ma60`, and the constant
  drift). Their point estimates are tiny (+0.09% to +0.19%), most win only
  6 of 9 folds, and the fold-level sign tests are not significant. `ma5` has
  the largest point estimate (+0.39%) but its interval includes zero.
- **The interval and the sign test disagree by construction.** The bootstrap
  is magnitude-weighted over days, while the sign test counts folds. With
  nine noisy years, a few large positive folds can pull the pooled interval
  away from zero without a consistent fold-level edge.
- **The joint linear model destroys the univariate signal.** The 13-feature
  OLS and the fixed-alpha ridge are negative on both windows even though many
  single-feature rules are positive. With 13 correlated inputs and a
  signal-to-noise ratio this low, estimation variance dominates.
- **Small combinations do not rescue it.** Top-k models selected on validation
  stay positive on test for k = 1 to 8, but every interval includes zero and
  no model wins more than 6 of 9 folds. The best simple rule is still not a
  statistically established edge.
- **The stored models fail to exploit what little signal exists.** DENFIS,
  XGBoost and LSTM are negative on test; DENFIS and LSTM are negative even in
  every leave-one-fold-out variant. This is consistent with over-fitting a
  weak signal, not with a coding error; the synthetic-signal check already
  shows the same pipeline learns a planted signal.

## Caveat for any follow-up

The test years have now been inspected in this diagnostic. A model designed
from these results cannot be sold as a clean test-set result. The defensible
route is to design it on validation only, freeze it, and evaluate once on the
untouched 2026 holdout.

## Reproduction

```bash
python scripts/feature_information_check.py
```
