## 3. Literature review
### Adaptive fuzzy models and evolving structure

A neuro-fuzzy model combines a fuzzy representation of local input regions with
parameters estimated from data. The fuzzy rules offer an explicit intermediate
representation: a prediction can be decomposed into rule activation and local
consequent output. This representation is useful only if the explanation names
its variables, scales, and assumptions. A large set of numeric rules is not
automatically interpretable, and an interpretable architecture does not guarantee
that its fitted rules describe an economically meaningful relationship.

Jang's ANFIS (1993) [1] is an early adaptive-network formulation for fuzzy inference.
It provides a useful reference for connecting membership functions, rule
activations, and consequent estimation within a learning system. Its importance
here is conceptual rather than empirical: ANFIS is not run as an additional
baseline. A distinction relevant to this project is whether model structure is
specified before training or can change as new input regions appear. Financial
features can change their distribution over time, motivating an investigation
of evolving structure without assuming that evolution will improve prediction.

Song and Kasabov's ECM manuscript (2001) [5] describes an incremental geometric
partition of an input stream. Its state consists of centres and radii rather
than a stored copy of all historical observations. A radius threshold controls
when the partition expands and when a new region is required. This makes ECM a
natural antecedent generator for local fuzzy predictors. The distance
normalisation is important: thresholds stated in a dimension-normalised metric
cannot be silently substituted into ordinary Euclidean distance. The implementation and
method notes specify the metric so that the grid is reproducible.

Kasabov and Song's DENFIS (2002) [6] couples evolving fuzzy structure with local
inference for time-series prediction. The original work covers online and
offline formulations and dynamically selected activated rules. The implementation adopts the
broader ECM/TSK/local-estimation approach, while following this project's fixed
Gaussian and weighted-RLS protocol. The implemented model is therefore described
as a DENFIS-inspired variant and list its deviations. The contribution relative to the established DENFIS
algorithm is an independently written,
bounded implementation and a controlled comparison of its frozen and adapting
forms on this dataset.

### Related evolving systems and modern directions

Angelov and Filev (2004) [7] study online identification of evolving Takagi–Sugeno
models. Their work places DENFIS within a wider family of systems that adapt
structure and local parameters from streams. Rong and colleagues' SAFIS (2006) [8]
uses rule influence in sequential structural adaptation. Pratama and colleagues' PANFIS
(2014) [9] provides another incremental fuzzy-learning architecture. These methods
motivate questions about model size, structural adaptation, and local learning;
comparative scores are not reported for implementations that were not run.

Recent deep neuro-fuzzy research expands the representational setting beyond
small tabular systems. Yeganejou and colleagues' DCNFIS preprint (2023, revised
2024) [11] investigates convolutional neuro-fuzzy modelling and explanations for
visual data. Its application domain differs substantially from daily financial
returns. It supports a future research direction, not a claim that its reported
accuracy transfers to DBS or that the model matches contemporary state of the
art. Alateeq and Pedrycz's 2024 logic-oriented fuzzy-neural survey [12] provides additional
review context, especially the relation between rule representation and
interpretation. Its discussion is distinct from the empirical return task.

### Baselines and scope of comparison

Chen and Guestrin's XGBoost (2016) [10] supplies a regularised nonlinear tree baseline.
Hochreiter and Schmidhuber's LSTM (1997) [4] supplies the recurrent modelling basis
for the temporal baseline. The repository's LSTM receives twenty-day sequences
of the thirteen features; DENFIS and XGBoost receive the current feature vector.
This difference expresses their inductive biases and must be disclosed. The three inputs are
not described as identical simply because their feature names
match. Hyperparameter budgets also differ and are listed in the shared protocol.

| Family | Structural idea | Role in this study |
|---|---|---|
| ANFIS | Adaptive-network fuzzy inference | Historical and architectural context |
| DENFIS / ECM | Evolving input regions and local inference | Basis of implemented method |
| eTS / SAFIS / PANFIS | Related sequential fuzzy adaptation | Literature context only |
| Deep neuro-fuzzy models | Learned representations with fuzzy components | Future direction only |
| XGBoost | Regularised boosted trees | Empirical baseline |
| LSTM | Recurrent sequence representation | Empirical baseline |

The evaluation question is deliberately narrow: under the same cached market
data, causal preprocessing, and chronological test windows, does this variant
produce useful return forecasts and a defensible long/flat strategy? Statistical
forecast comparison follows Diebold and Mariano (1995) [3], while dependent-return
uncertainty is estimated with the stationary bootstrap of Politis and Romano
(1994) [2]. These tools quantify uncertainty; neither converts an exploratory
experiment into proof of future trading profitability.
