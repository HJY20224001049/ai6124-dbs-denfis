# What the literature says we can do about these problems

This note maps the three verified problems to established solutions. It is a
reading guide, not a new empirical result. The project's method references stay
in `docs/refs.md`; this file covers forecast evaluation, shrinkage, forecast
combination, regularisation and volatility forecasting.

## The problem in one sentence

The signal in daily DBS returns is weak, the nine-year test window cannot
resolve effects below about 0.5% RMSE, and the 13-feature joint models
overfit what little signal exists. No data or implementation defect was found.

## Mapping

| Problem | Literature solution | Key references | What our evidence says |
|---|---|---|---|
| Forecasts lose to the zero forecast | Shrink forecasts toward the historical mean or zero | Campbell & Thompson (2008); Welch & Goyal (2008) | A validation-fitted shrinkage lifts DENFIS from −0.76% to −0.07%, still below zero. Helpful, not sufficient. |
| Single models are unstable | Combine forecasts; prefer equal weights over estimated weights | Bates & Granger (1969); Timmermann (2006); Stock & Watson (2004); Rapach, Strauss & Zhou (2010); Smith & Wallis (2009) | A 50/50 blend beats XGBoost in only 3/9 folds; 5/9 folds choose w = 0. The models are too correlated for combination to help. |
| 13 correlated features destroy the signal | Shrinkage and variable selection: ridge, lasso, elastic net, PCR/PLS | Hoerl & Kennard (1970); Tibshirani (1996); Zou & Hastie (2005); Gu, Kelly & Xiu (2020) | 13-feature OLS/ridge are negative on validation and test; top-k simple models stay positive but every interval spans zero. The direction is right; the signal is too weak. |
| The sample cannot resolve small effects | Longer history, panel data, or a lower-variance target | Welch & Goyal (2008); Gu, Kelly & Xiu (2020); Inoue & Kilian (2004) | The 80%-power MDE is 0.50% for the best univariate rule and 1.10% for the DENFIS–XGBoost gap. A 0.4% edge needs roughly 15 years of daily data. |
| The requirement mixes point estimates and significance | Pre-specify the test; use the model confidence set and multiple-testing corrections | Diebold & Mariano (1995); Clark & West (2007); Giacomini & White (2006); Hansen, Lunde & Nason (2011); White (2000); Hansen (2005); Romano & Wolf (2005) | Our pairwise DM tests and fold-level sign tests disagree. A model confidence set would report which models are statistically indistinguishable instead of ranking noise. |
| Accuracy and economic value are conflated | Evaluate statistical and economic significance separately | Pesaran & Timmermann (1995); Cenesizoglu & Timmermann (2012); Kandel & Stambaugh (1996); Bossaerts & Hillion (1999) | DENFIS has the best point-estimate Sharpe but a CI spanning zero and a lower return than buy-and-hold. The two claims should not be merged. |
| Daily returns carry too little signal | Change the target to realised volatility or a longer horizon | Corsi (2009); Bollerslev (1986); Hansen, Huang & Shek (2012); Campbell & Shiller (1988); Fama & French (1988); Hansen & Hodrick (1980) | Ridge predicts 20-day realised volatility 20.9% better than RV20 persistence (R² = 0.37). This is the only large, robust signal in the project. It changes the task. |
| Model uncertainty is ignored | Bayesian shrinkage and model averaging | Avramov (2002); Cremers (2002) | A principled version of the shrinkage we already tried; not an independent route to a positive return result. |
| Bagging should stabilise the model | Bagging unstable learners | Breiman (1996) | Bagging improves 12–19 of 20 grid points, but validation-based selection already avoids those unstable points; at the selected point it gains +0.07%. |
| The evolving fuzzy model is too complex | Parsimony and interpretability are the method's comparative advantage | Kasabov & Song (2002); Angelov & Filev (2004); Lughofer (2008); Pratama et al. (2014) | Consistent with simplifying DENFIS (fewer rules, fewer features). It does not create a statistical edge over a tuned GBM. |

## What I would actually do, in order

1. **Lowest risk: report the negative result with the three diagnostics.**
   The literature is clear that failing to beat the random walk on daily
   returns is common, not embarrassing (Meese & Rogoff, 1983; Welch & Goyal,
   2008; Campbell & Thompson, 2008). The report should say "no resolvable
   edge", not "the model is worse".

2. **Best chance of a positive result with integrity: pre-register a simple
   model and test it once on the 2026 holdout.** Select one to three features
   on validation only (for example `ma5`, or a ridge on the top-k validation
   features), freeze it before 15 November, and evaluate it once on the
   untouched 2026 data. This follows the regularisation and shrinkage
   literature (Hoerl & Kennard, 1970; Tibshirani, 1996; Campbell & Thompson,
   2008; Gu, Kelly & Xiu, 2020) and avoids the test-set selection problem.

3. **Strongest signal, but a different task: volatility.** Fit the same
   DENFIS or a regularised linear model to 20-day realised volatility instead
   of the return. The HAR and realised-GARCH literature predicts a much larger
   R² (Corsi, 2009; Hansen, Huang & Shek, 2012), and our pilot already shows
   ridge beats RV20 persistence by 20.9%. This needs the instructor's written
   approval because it changes the assignment.

4. **Power fix, but time-consuming: extend the sample.** DBS has a longer
   history than 2010. Extending the cache back to the 1990s would lower the
   MDE by roughly the square root of the sample ratio, but it changes the
   folds and requires a full re-run. Not realistic before 15 November.

5. **Evaluation fix, cheap and useful now: report a model confidence set and
   the MDE.** Replace "DENFIS is 0.47% worse" with "DENFIS and XGBoost are
   statistically indistinguishable; the study can only resolve gaps above
   about 1.1%". This is the honest version of the professor's ordering
   requirement (Hansen, Lunde & Nason, 2011; Inoue & Kilian, 2004).

## What not to do

- **Do not add capacity.** More rules, type-2 sets, kernel RLS, bagging or a
  deeper network all increase variance. The literature that promotes them
  compares them with other fuzzy models, not with a tuned GBM on daily equity
  returns, and our validation diagnosis points the other way.
- **Do not switch to XGBoost or LSTM as the main model.** Neither beats the
  zero forecast; the switch would abandon the method line and still fail the
  requirement.
- **Do not choose a model or a feature on the 2017–2025 test years.** Those
  years have now been inspected in the diagnostics. The clean test is the 2026
  holdout.
- **Do not report the best fold or the best grid point as an improvement.**
  The six rejected routes already show how easy that is; White (2000) and
  Hansen (2005) are the standard references for why it is invalid.

## Verified source notes

- **Negative out-of-sample R² is common, not proof of no signal.** Welch &
  Goyal (2008) find most equity-premium predictors have low or negative
  out-of-sample R². Zevallos & Rubesam (2026) derive the finite-sample
  distribution of the Campbell–Thompson R² and show it is downward biased:
  negative values arise frequently under weak but genuine predictability,
  especially with short samples and many predictors. Our −0.76% pooled R² is
  therefore consistent with "weak signal", not only with "no signal".
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0165176526002053)
- **Sign restrictions and shrinkage help.** Campbell & Thompson (2008) show
  that imposing the theoretically expected coefficient sign and a non-negative
  equity-premium forecast makes many regressions beat the historical average;
  the R² is small but economically meaningful. Our validation-fitted shrinkage
  lifts DENFIS to −0.07%, still below zero, but the direction is right.
  [RFS](https://academic.oup.com/rfs/article-abstract/21/4/1509/1567518)
- **Combination works when individual models are unstable.** Rapach, Strauss &
  Zhou (2010) report a mean combination forecast with roughly +3.6% OOS R²
  while most individual equity-premium predictors are negative. Smith & Wallis
  (2009) explain the "combination puzzle": equal weights often beat estimated
  weights because estimated weights add finite-sample noise. Our blend fails
  for a different reason — the models are too correlated, and 5/9 folds choose
  w = 0.
  [RFS](https://academic.oup.com/rfs/article-abstract/23/2/821/1604687) ·
  [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0084.2008.00541.x)
- **Regularisation and dimension reduction are the standard fix for many
  correlated predictors.** Gu, Kelly & Xiu (2020) report negative OOS R² for
  OLS with 900+ characteristics, about 0.26% monthly R² for ridge/lasso/
  elastic net, and 0.33–0.40% for nonlinear models. Their message is that
  penalisation, not capacity, does the work. Our 13-feature OLS/ridge are
  negative; top-k simple models are positive but not significant.
  [RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276)
- **The zero benchmark is a random walk, and it is hard to beat.** Meese &
  Rogoff (1983) established the classic result for exchange rates; the same
  "random-walk benchmark" logic applies to daily prices. Failing to beat it is
  a well-known outcome, not an implementation error.
  [Springer chapter on the Meese–Rogoff puzzle](https://rd.springer.com/chapter/10.1057/9781137452481_1)
- **The right test against a nested benchmark is Clark–West, not plain DM.**
  Clark & West (2007) show that the larger model's estimated coefficients
  inflate its MSPE in finite samples, so the standard DM test is biased
  against it. The MSPE-adjusted one-sided test is the standard for comparing a
  model with a driftless random walk.
  [PDF](https://users.ssc.wisc.edu/~kwest/publications/2000/Approximately%20Normal%20Tests%20for%20Equal%20Predictive%20Accuracy%20in%20Nested%20Models.pdf)
- **DM needs small-sample care.** Harvey, Leybourne & Newbold (1997) give the
  DM* correction; Harvey, Leybourne & Whitehouse (2017) document negative
  long-run variance estimates in small samples; Coroneo & Iacone (2021) show
  power falls as loss-differential autocorrelation rises. Our nine-fold,
  daily-loss design is exactly the regime where these corrections matter.
  [IJF](https://www.sciencedirect.com/science/article/abs/pii/S0169207017300559)
- **Report a model confidence set, not a ranking of noise.** Hansen, Lunde &
  Nason (2011) construct a set that contains the best model with a given
  confidence and explicitly allows for uninformative data. White (2000) and
  Hansen (2005) control data snooping across many models. A MCS on our four
  models would likely contain DENFIS, XGBoost and LSTM.
  [Econometrica](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA5771) ·
  [White PDF](https://users.ssc.wisc.edu/~bhansen/718/White2000.pdf)
- **Volatility is the target with a large, robust signal.** Corsi (2009) HAR
  models; Corsi, Pirino & Reno (2010) and Duong & Swanson (2015) find log
  specifications beat levels out of sample; the GARP HAR guide reports up to
  ~24% QLIKE reduction from WLS-HAR versus OLS-HAR. Our ridge already beats
  RV20 persistence by 20.9% (R² = 0.37) on validation. This is the strongest
  route, but it changes the assignment.
  [GARP guide](https://www.garp.org/hubfs/Whitepapers/a2r1W000000iDb0QAE_RiskIntell.6.20.19.Whitepaper.Volatility.pdf)
- **There is no head-to-head DENFIS-versus-XGBoost stock-return study in the
  indexed literature.** DENFIS has been compared with ANN, ANFIS, LSTM, ARIMA
  and other fuzzy/evolving systems; Vella & Ng (2014) compare ANN, ANFIS and
  DENFIS on 5-minute LSE data and find DENFIS is not the winner. This supports
  two claims: our comparison fills a gap, and adding EFS capacity is not
  evidence-backed as a way to beat a tuned GBM.
  [Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231214004470)

## References

- Avramov, D. (2002). Stock return predictability and model uncertainty.
  *Journal of Finance*, 57(5), 2329–2364.
- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts.
  *Operational Research Quarterly*, 20(4), 451–468.
- Bossaerts, P., & Hillion, P. (1999). Implementing statistical criteria to
  select return forecasting models: What do we learn? *Review of Financial
  Studies*, 12(2), 405–428.
- Bollerslev, T. (1986). Generalized autoregressive conditional
  heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
- Campbell, J. Y., & Shiller, R. J. (1988). Stock prices, earnings, and
  expected dividends. *Journal of Finance*, 43(3), 661–676.
- Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns
  out of sample: Can anything beat the historical average? *Review of
  Financial Studies*, 21(4), 1509–1531.
- Cenesizoglu, T., & Timmermann, A. (2012). Do return prediction models add
  economic value? *Journal of Banking & Finance*, 36(11), 2974–2987.
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal
  predictive accuracy in nested models. *Journal of Econometrics*, 138(1),
  291–311.
- Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Cremers, K. J. M. (2002). Stock return predictability: A Bayesian model
  selection perspective. *Review of Financial Studies*, 15(4), 1223–1249.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Fama, E. F., & French, K. R. (1988). Dividend yields and expected stock
  returns. *Journal of Financial Economics*, 22(1), 3–25.
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability.
  *Econometrica*, 74(6), 1545–1578.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
  learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of
  Business & Economic Statistics*, 23(4), 365–380.
- Hansen, P. R., Huang, Z., & Shek, H. H. (2012). Realized GARCH: A joint
  model for returns and realized measures of volatility. *Journal of Applied
  Econometrics*, 27(6), 877–906.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
  *Econometrica*, 79(2), 453–497.
- Hansen, L. P., & Hodrick, R. J. (1980). Forward exchange rates as optimal
  predictors of future spot rates: An econometric analysis. *Journal of
  Political Economy*, 88(5), 829–853.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation
  for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Inoue, A., & Kilian, L. (2004). In-sample or out-of-sample tests of
  predictability: Which one should we use? *Econometric Reviews*, 23(4),
  371–402.
- Kandel, S., & Stambaugh, R. F. (1996). On the predictability of stock
  returns: An asset-allocation perspective. *Journal of Finance*, 51(2),
  385–424.
- Meese, R. A., & Rogoff, K. (1983). Empirical exchange rate models of the
  seventies: Do they fit out of sample? *Journal of International Economics*,
  14(1–2), 3–24.
- Pesaran, M. H., & Timmermann, A. (1995). Predictability of stock returns:
  Robustness and economic significance. *Journal of Finance*, 50(4),
  1201–1228.
- Rapach, D. E., Strauss, J. K., & Zhou, G. (2010). Out-of-sample equity
  premium prediction: Combination forecasts and links to the real economy.
  *Review of Financial Studies*, 23(2), 821–862.
- Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalized
  data snooping. *Econometrica*, 73(4), 1237–1282.
- Smith, J., & Wallis, K. F. (2009). A simple explanation of the forecast
  combination puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3),
  331–355.
- Stock, J. H., & Watson, M. W. (2004). Combination forecasts of output growth
  in a seven-country data set. *Journal of Forecasting*, 23(6), 405–430.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
  *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J.
  Granger & A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 1,
  pp. 135–196). Elsevier.
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical
  performance of equity premium prediction. *Review of Financial Studies*,
  21(4), 1455–1508.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5),
  1097–1126.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the
  elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2),
  301–320.
