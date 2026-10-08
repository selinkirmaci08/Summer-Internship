# Plot Selection Conditions

The condition-to-plot table EDA step 10 applies. Every row other than the "always" rows is gated, so plot generation stays proportional to what the data actually surfaced rather than to how wide the dataset is.

Used by: `eda.md` step 10.

---

| Condition | Plot | Purpose |
|---|---|---|
| Classification task (always) | Class balance bar chart | Confirms imbalance severity, informs metric choice |
| Regression task (always) | Target distribution histogram | Shows skew/outliers in the target |
| Target balance differs notably by segment | Grouped bar chart: target rate by segment | Surfaces subgroup bias |
| Datetime column present (always) | Row count over time | Reveals collection gaps or volume shifts |
| Datetime + target present | Target rate over time | Checks for concept drift |
| Any numeric feature (always) | Histogram per numeric column | Low-cost baseline coverage |
| Feature in `top_features`, numeric | Scatter or binned mean-target-rate vs. target | Reveals nonlinear/threshold relationships correlation misses |
| High outlier count for a feature | Box plot | Outlier check, pairs with `out_of_range_values` |
| ≥2 numeric features (always) | Correlation heatmap | Standard reference for feature relationships |
| Pair in `multicollinearity.high_pairs` | Scatter of that pair | Confirms redundancy before recommending a drop |
| ≤6–8 features in `top_features` | Pairplot of just those features | Spot clusters/multicollinearity — never run on all columns |
| Categorical feature in `top_features` or `high_cardinality` | Bar chart, top-N categories only | Shows dominant vs. long-tail without rendering every category |
| Categorical feature correlates with target | Grouped bar/box: target rate by category | Categorical equivalent of the numeric scatter |
| Column missing_pct above threshold (e.g. >5%) | Missingness matrix/heatmap | Shows co-missing patterns across columns |
| Entry in `correlated_missingness_pairs` | Annotated highlight of that pair | Confirms joint-missingness before joint handling |
| `out_of_range_values` found for a feature | Box plot with flagged points marked | Distinguishes genuine outliers from data entry errors |
