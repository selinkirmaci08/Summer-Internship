# Cross-Validation Strategy Selection

Source: [scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).

Choose the CV scheme *before* tuning hyperparameters or picking a final metric — the wrong scheme silently invalidates every score computed downstream (most commonly: ordinary K-Fold used on grouped or time-ordered data, which leaks information across folds).

## 1. Decision table

| Data condition | Recommended strategy | Alternative | Avoid |
|---|---|---|---|
| Balanced, i.i.d. data, no special structure | K-Fold | ShuffleSplit | — |
| Classification with class imbalance | **StratifiedKFold** | StratifiedShuffleSplit | Plain K-Fold / ShuffleSplit |
| Grouped/clustered data (e.g. multiple rows per patient/customer/device) | **GroupKFold** | StratifiedGroupKFold (if also imbalanced) | K-Fold, StratifiedKFold |
| Grouped data + class imbalance | **StratifiedGroupKFold** | GroupShuffleSplit | K-Fold |
| Time series / temporal data | **TimeSeriesSplit** | — | K-Fold, ShuffleSplit, StratifiedKFold (all leak future into past) |
| Very small dataset | Leave-One-Out or RepeatedKFold | StratifiedKFold | Standard K-Fold (high variance estimate) |
| Large dataset, need efficiency / flexible train-test ratio | ShuffleSplit | K-Fold | Leave-One-Out (too expensive) |
| Explicit leave-one-experiment/time-period-out evaluation | Leave-One-Group-Out (LOGO) | GroupKFold | — |
| Many groups, want random sampling of group partitions | Group Shuffle Split | LeavePGroupsOut | — |

## 2. Strategy reference

| Strategy | What it does |
|---|---|
| K-Fold | Splits into k equal folds; trains on k−1, tests on the remainder; repeats k times |
| StratifiedKFold | K-Fold that preserves each fold's class proportions — default for classifiers |
| Repeated (Stratified) K-Fold | Runs K-Fold multiple times with different random splits, for a more stable variance estimate |
| Leave-One-Out (LOO) | k = n; trains on all but one sample per fold. Maximizes training data but is expensive and high-variance — prefer 5/10-fold K-Fold unless n is genuinely tiny |
| ShuffleSplit | Randomly shuffles then splits repeatedly; independent splits with full control over train/test proportions and iteration count |
| GroupKFold | Ensures a given group never appears in both train and test |
| StratifiedGroupKFold | GroupKFold + class-balance preservation |
| Leave-One-Group-Out (LOGO) | Each split leaves out all samples from exactly one group |
| Group Shuffle Split | Random shuffle/split at the group level — efficient alternative to exhaustive group leave-out when there are many groups |
| TimeSeriesSplit | Forward-chaining splits: each training set is a superset of the previous one, and the test fold is always *later in time* than training. Supports a `gap` parameter to leave a buffer between train and test — use this to simulate a realistic forecast lead time instead of testing on the instant right after training ends. Called **walk-forward validation** in the forecasting literature — same concept, different name (source: [meegle.com](https://www.meegle.com/en_us/topics/fine-tuning/fine-tuning-for-time-series-analysis)); use it during hyperparameter tuning as well as final evaluation, not just the latter — see `hyperparameter_tuning.md` §5. |

## 3. Practical rules

- **Never shuffle time-ordered data.** `TimeSeriesSplit` doesn't shuffle by design — respect that.
- **Fit any data-dependent transform (scaler, imputer, encoder) only on the training fold**, then apply it to the validation fold. Wrap the model in a `Pipeline` so `cross_val_score`/`cross_validate`/`GridSearchCV` do this automatically per fold.
- **Use `cross_validate()` instead of `cross_val_score()`** when you want multiple metrics computed per fold.
- Default in scikit-learn's own `cross_val_score` for a classifier with an integer `cv` is already `StratifiedKFold` — but be explicit about it when building a pipeline for the training skill, rather than relying on the default.
- If a task involves both hyperparameter tuning and a final held-out performance estimate, keep a dedicated evaluation split untouched by CV/tuning (development set vs. evaluation set), or use nested CV.
