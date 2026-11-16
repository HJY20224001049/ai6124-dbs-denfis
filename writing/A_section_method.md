## 5. Method

### 5.1 Evolving clustering (ECM)

The evolving clustering method constructs rule regions incrementally from a stream of standardised inputs, rather than specifying all rule locations beforehand [5]. Each rule maintains a centre \(c_i\) and radius \(R_i\), and the implementation measures distance as \(D_i(x)=\lVert x-c_i\rVert_2/\sqrt{d}\), where \(d\) is the input dimension. An input covered by an existing radius increases that region's support without changing its geometry. Otherwise, the algorithm selects the minimum extended distance \(S_i=D_i+R_i\); when this exceeds \(2Dthr\), it creates a new centre at the input with initial radius zero, subject to the rule limit. For an admissible expansion, the radius becomes \(S_i/2\) and the centre moves towards the input so that its distance to that input equals the new radius. The threshold \(Dthr\) therefore sets the resolution at which the input space is partitioned.

Two deliberate engineering constraints distinguish the implemented rule base from the unbounded construction: a hard limit of 200 rules and optional merging based on support. When normalised centre distance falls below \(merge\_factor\cdot Dthr\), the merge retains the higher-support rule and combines the support counts. Its radius is enlarged to contain the removed region, while the surviving consequent is retained. These choices control the size and representation of the rule base; they are not presented as methodological improvements to the original algorithm. These constraints identify the bounded rule base actually evaluated and do not imply that the source method requires the same restrictions. The implementation differences are documented in `docs/denfis_notes.md`, so the stated model can be distinguished from the source method.

### 5.2 Gaussian membership and first-order TSK inference

Each coordinate of a rule has a Gaussian membership centred at its corresponding centre coordinate, and the effective coordinate width is \(\sigma_i=\sqrt{d}\max(R_i,0.1Dthr)\) for the configured thresholds. The dimension factor follows from computing activation with a mean squared coordinate distance in the code, rather than an unnormalised sum. The memberships and normalised rule strengths are
\[
\mu_{ij}(x_j)=\exp\!\left[-\frac{(x_j-c_{ij})^2}{2\sigma_i^2}\right],\qquad
w_i(x)=\prod_{j=1}^{d}\mu_{ij}(x_j),\qquad
h_i(x)=\frac{w_i(x)}{\sum_k w_k(x)}.
\]
The strengths sum to one, and the prediction \(\hat y(x)=\sum_i h_i(x)a_i^\top\phi(x)\) is a convex combination of the local linear outputs, with \(\phi(x)=[1,x^\top]^\top\). Normalisation keeps the combination within the range of those outputs, including when an input is distant from the centres; the separate numerical safeguard described below bounds the final forecast.

A first-order Takagi–Sugeno consequent represents each local output as a linear function of the input, following the local inference structure associated with DENFIS [6]. Its coefficients allow the fitted relationship to vary with the coordinates within a rule region. A zero-order consequent instead supplies a constant for that region, so changing an input without changing the rule weights cannot change that local output. The first-order choice is motivated by representing slowly varying local return relationships and extending a local trend within a region. This is a modelling rationale for the chosen consequent form, rather than an assumption that the relationship must be useful in every evaluation period.

### 5.3 Weighted recursive least squares

The consequent of each rule is estimated by weighted recursive least squares, with its normalised activation \(h_i\) determining the contribution of the current observation. For the augmented input \(\phi=[1,x^\top]^\top\), coefficient vector \(a_i\), covariance matrix \(P_i\), and realised label \(y\), the updates are
\[
K_i=\frac{h_iP_i\phi}{\lambda+h_i\phi^\top P_i\phi},\qquad
a_i\leftarrow a_i+K_i(y-\phi^\top a_i),\qquad
P_i\leftarrow\frac{P_i-K_i\phi^\top P_i}{\lambda}.
\]
The residual in this expression is the error of the individual rule's linear output. The forgetting factor \(\lambda\) controls the retention of earlier information, and setting it to one removes forgetting. With a single rule, \(h_1=1\), so the update also becomes unweighted RLS.

The covariance initialisation is \(100I\), and numerical checks protect the recursive calculation from an unsuitable state. A reset is triggered by nonfinite values, a covariance trace exceeding \(10^8\), or a nonpositive covariance diagonal entry. These checks describe the implemented safeguards, rather than a complete positive-definiteness test. The affected covariance is reset to \(100I\). The final prediction is clipped to the log-return interval \([-0.2,0.2]\), limiting the effect of an ill-conditioned update on the reported forecast. This clipping is a numerical protection applied to the fitted predictor, rather than a conclusion about which extreme returns can occur.

### 5.4 Frozen versus online

The frozen and online variants share the same model structure and training procedure, with their difference arising during test-period operation. In the frozen evaluation, the final training window supplies the fitted model that is subsequently used without parameter updates. In the online evaluation, the current forecast must be recorded before its realised target is supplied to the learner. Reversing these operations would allow an observation's own answer to influence its forecast. Contract tests check that calling `predict` leaves the model state unchanged, making prediction distinct from an explicit update through `partial_fit`.

- **Frozen:** `denfis_frozen` is fitted once on the final training window and remains fixed throughout the test year.
- **Online:** `denfis_online` predicts and records the output first, then calls `partial_fit(x, y)` after the corresponding label has been realised.

### 5.7 The final configuration: three one-rule components

B2 candidates were scored only on validation years: for evaluation year \(Y\), selection training uses labels through \(Y-2\) and the validation score uses \(Y-1\). The candidate-selection run did not index the designated test windows or open the final holdout. However, the historical test years had already been examined during earlier diagnostics, as disclosed in Section 8.5. Consequently, selection on validation years does not make the full research process a clean independent confirmation. Keeping these roles explicit distinguishes the evidence used to choose a setting from the evidence used to assess the selected model. The final holdout supplies the separate confirmation, with its outcome reported in Section 7 rather than used to choose this configuration.

B2 is the equal-weight average of three one-rule DENFIS components, respectively using `ma5`, Bollinger percent B (`boll_b`), and the last return (`ret_0`). Each component additionally receives `is_exdiv` and `is_post_exdiv_1`, giving three input coordinates per component. The settings are fixed at \(Dthr=3.0\), \(\lambda=1.0\), and `max_rules=1`. The simplification was motivated by the earlier diagnostic observation that the full 13-feature configuration did not beat the zero forecast in the already-inspected test years. Validation diagnostics showed a flat rule-count curve over one to ten rules, with four rules less favourable. The specific B2 configuration was selected on validation years, while the historical diagnostic motivation remains explicitly disclosed.

For a one-rule component the normalised activation is identically one, so each component is a least-squares linear fit on an intercept, its selected feature, and the event dummies. Initial covariance \(100I\) supplies a weak \(\ell_2\) prior: with \(\lambda=1\), the recursion terminates at \(a=(\Phi^\top\Phi+0.01I)^{-1}\Phi^\top y\). In the project training windows its difference from ordinary least squares is below \(1.5\times10^{-6}\) times the target standard deviation, so the prior is numerically small but nonzero. The fuzzy layer provides normalisation and weighting in this limit, while the three components average the separate fitted feature relationships. This interpretation can therefore be checked against a directly computed linear solution, rather than inferred solely from the model name. Numerical tests in `tests/test_denfis_contract.py` check predictions against the closed form to \(10^{-12}\) and the difference from OLS against \(10^{-4}\) times the target standard deviation.

The preregistered adoption rule required a mean validation improvement of at least 0.05 percentage points without reducing the number of positive folds. Sixteen one-factor variants considered component counts, event encodings, averaging weights, shrinkage, rolling windows, gap or intraday components, inner calibration, and reference-model mixing. The variants included two, three, five or seven components; mean, median or inverse-variance weights; shrinkage by 0.75 or 1.25; and three- or five-year windows. None met that rule, so B2 was retained. Its mean validation RMSE skill is **+0.497%**, with **8/9** positive folds, pooled skill **+0.480%**, and a 95% interval of **[+0.161%, +0.780%]**. These figures describe validation-year selection evidence, and the test and holdout results are reported in Section 7.
