# Model Selection — Time Series Forecasting

Sources: [machinelearningmastery Time Series Forecasting Methods Cheat Sheet](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/), [Time Series Forecasting Part 2: Selecting Algorithms (Data Science at Microsoft)](https://medium.com/data-science-at-microsoft/time-series-forecasting-part-2-of-3-selecting-algorithms-11b6635f61bb), [5 Must-Know Techniques for Mastering Time-Series Analysis](https://towardsdatascience.com/5-must-know-techniques-for-mastering-time-series-analysis-a23ccf4d053a/), [Training a Time-Series Forecasting Model Using Automated Machine Learning](https://techcommunity.microsoft.com/blog/educatordeveloperblog/training-a-time-series-forecasting-model-using-automated-machine-learning/4212384) (pasted directly by the user — WebFetch was blocked on this one twice), [10 Incredibly Useful Time Series Forecasting Algorithms](https://www.advancinganalytics.co.uk/blog/2021/06/22/10-incredibly-useful-time-series-forecasting-algorithms), [Fine-Tuning for Time Series Analysis](https://www.meegle.com/en_us/topics/fine-tuning/fine-tuning-for-time-series-analysis) (tuning-specific — see `hyperparameter_tuning.md`).

**Resolved — AR/ARIMA seasonality framing:** the Microsoft TechCommunity article described plain AR as suited to data with "a clear trend or seasonality," and ARIMA as handling "trends and seasonality." The machinelearningmastery cheat sheet disagreed on both counts (AR: no trend/seasonality; ARIMA: trend but not seasonality — that's what SARIMA adds). Per your call, §2 below follows the **machinelearningmastery cheat sheet** on both points — decided 2026-08-21.

Time series is fundamentally different from the classical-ML model families in `model_selection_supervised.md`: this file covers models built to use temporal structure directly (autoregression, seasonality, trend) rather than treating each row as an independent observation.

## 1. Two competing approaches — decide this first

| Approach | What it means | When to prefer |
|---|---|---|
| **Classical statistical forecasting** (§2 below) | Fit a model (ARIMA, exponential smoothing, etc.) directly to the time-ordered series, using its own past values/errors/seasonality as predictors | Single or few series, need interpretable trend/seasonality decomposition, limited exogenous data, forecasting is the whole task |
| **ML-as-regression on engineered features** | Turn the forecasting problem into a tabular regression problem using lag/rolling/calendar features (see shared preprocessing reference §5), then apply any model from `model_selection_supervised.md` (tree ensembles and gradient boosting are the common default here) | Many parallel series, rich exogenous/external features, need to reuse the same training pipeline as non-time-series problems, or trend/seasonality are complex/non-linear. *(XGBoost specifically confirmed as a recognized time-series approach by [advancinganalytics.co.uk](https://www.advancinganalytics.co.uk/blog/2021/06/22/10-incredibly-useful-time-series-forecasting-algorithms) — same gradient-boosting family already covered in `model_selection_supervised.md` §6, just fed lag/rolling features instead of arbitrary tabular ones.)* |
| **Automatic algorithm selection** (Auto ARIMA, Facebook Prophet, AutoML forecasting frameworks) | Let a tool search over classical model variants (and sometimes ML variants) and pick the best by backtested accuracy | Don't want to hand-tune (p,d,q)(P,D,Q) orders; need a fast, reasonable default before investing in a bespoke model. Prophet specifically is called out (Microsoft TechCommunity article) as strong when seasonal effects dominate — it's built to decompose trend + multiple seasonalities + holiday effects with minimal tuning, which makes it a common first choice for business/calendar-driven forecasts. Still thin coverage overall — treat as a starting point to search further, not a full reference. |

Both non-automatic approaches still require the CV and metric choices in §3–4 below.

**Pipeline rule:** a time-series run must fit *both* non-automatic approaches at least once before
it may stop — see `../training.md` step 2b-ts, and the matching coverage gates in the Planner and
Diagnosis. The automatic row is a reference point, not a substitute for either.

**Out of scope:** deep-learning sequence models for forecasting — LSTM, Amazon's DeepAR (probabilistic autoregressive RNN), N-BEATS (residual-link forecasting network), and Google's Temporal Fusion Transformer (attention-based, multi-horizon, interpretable) — are named by [advancinganalytics.co.uk](https://www.advancinganalytics.co.uk/blog/2021/06/22/10-incredibly-useful-time-series-forecasting-algorithms) as the current modern landscape beyond classical statistics. Consistent with this reference set's classical-ML scope (deep learning is excluded the same way MLPs are in `model_selection_supervised.md`), none of them are covered in depth here — flag them by name only if a task explicitly calls for deep learning, rather than re-researching from scratch.

**Operational caveat — concept drift:** [meegle.com](https://www.meegle.com/en_us/topics/fine-tuning/fine-tuning-for-time-series-analysis) flags that "sudden changes in data trends can render models ineffective" — a time series model's fit degrades as the underlying process shifts (regime changes, new seasonality, structural breaks), in a way that a one-time train/test split won't catch. This is more of a deployment/monitoring concern than a training-time one, but worth flagging to whatever stage owns model monitoring/retraining triggers downstream of this skill.

## 2. Classical forecasting model reference

| Model | Handles trend? | Handles seasonality? | Multivariate? | Exogenous vars? | Key parameters | Python (statsmodels) |
|---|---|---|---|---|---|---|
| Autoregression (AR) | No | No | No | No | order `p` | `tsa.ar_model.AutoReg` |
| Moving Average (MA) | No | No | No | No | order `q` | `tsa.arima.model.ARIMA(order=(0,0,q))` |
| ARMA | No | No | No | No | `p`, `q` | `tsa.arima.model.ARIMA(order=(p,0,q))` |
| ARIMA | **Yes** | No | No | No | `p`, `d` (differencing), `q` | `tsa.arima.model.ARIMA` |
| Seasonal ARIMA (SARIMA) | Yes | **Yes** | No | No | `(p,d,q)(P,D,Q)m`, `m`=seasonal period | `tsa.statespace.sarimax.SARIMAX` |
| SARIMAX | Yes | Yes | No | **Yes** | SARIMA params + exogenous array | `tsa.statespace.sarimax.SARIMAX` |
| Vector Autoregression (VAR) | No | No | **Yes** | No | order `p` | `tsa.vector_ar.var_model.VAR` |
| VARMA | No | No | Yes | No | `p`, `q` | `tsa.statespace.varmax.VARMAX` |
| VARMAX | No | No | Yes | Yes | VARMA params + exogenous | `tsa.statespace.varmax.VARMAX` |
| Simple Exponential Smoothing (SES) | No | No | No | No | smoothing level (auto-optimized) | `tsa.holtwinters.SimpleExpSmoothing` |
| Holt-Winters (HWES / triple exponential smoothing) | **Yes** | **Yes** | No | No | level/trend/seasonal smoothing params (auto-optimized) | `tsa.holtwinters.ExponentialSmoothing` |

**Decision shortcut:** start from what the series actually has — no trend/no seasonality → AR/MA/ARMA or SES; trend only → ARIMA; trend + seasonality → SARIMA or Holt-Winters; need external predictors (weather, promotions, price) on top of any of the above → SARIMAX; multiple related series modeled jointly → VAR/VARMA/VARMAX.

**Pipeline rule:** the classical slot in §1 is closed by a member of the ARIMA family above —
ARIMA, SARIMA or SARIMAX — with its orders searched and recorded, never by exponential smoothing
alone. When the shortcut points past plain ARIMA, plain ARIMA is still fitted once alongside the
chosen member, as the ablation that prices the seasonal/exogenous terms. See `../training.md`
step 2b-ts.

## 3. Validation

Use `TimeSeriesSplit` as covered in `cross_validation.md` — see that file's §on time series for the `gap` parameter and forward-chaining detail, which matters more for forecasting than for other regression problems (you're usually simulating a real forecast lead time, not just avoiding leakage).

## 4. Evaluation metrics

See `evaluation_metrics.md` §4 (Time Series / Forecasting Metrics) for the MAE vs. RMSE vs. SMAPE tradeoff pulled from these sources.

## 5. Feature engineering

Lag features, rolling statistics, calendar features, differencing, cyclic encoding, and stationarity testing belong to the preprocessing stage, not here — see `../../shared-references/preprocessing_and_feature_engineering.md` §5.
