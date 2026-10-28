# Agentic AutoML

Industrial internship at [Element61](https://www.element61.be), KU Leuven, summer 2026 (H04I9A).
Selin Kırmacı, supervised by Jeremy Matthynssens and Chidi Nweke (Element61) and Prof. Tom Van Cutsem (KU Leuven).

This repository holds an automated machine learning pipeline driven by LLM agents. Given only a dataset and a goal, it carries out the whole workflow the way a data scientist would. It explores the data, researches the domain, prepares the data, builds features, trains models, diagnoses the results and decides what to try next. Conventional AutoML tools only search over models. This pipeline also reasons about the data at every step.

The pipeline was tested on a real customer dataset: the daily bread sales of a Belgian bakery chain, forecast for one branch at six horizons (t+1, t+7, t+14, t+30, t+60 and t+90 days).

![The Agentic AutoML pipeline](graph/Pipeline%20Graph-selection_v2.png)

## How it works

Every stage of the workflow is a **skill**. A skill is a Markdown file that defines one responsibility, the inputs it may read, the output it must write and the rules it must respect, together with reference notes on best practices. Each skill runs as its own [Claude Code](https://docs.claude.com/en/docs/claude-code) agent.

| Skill | What it does |
|---|---|
| `planner` | Starts every run and decides which skill runs next, based on what earlier steps found. It holds the stopping criteria and asks the user when a skill raises a question it cannot answer. |
| `eda` | Analyses every source table, judges whether the goal is attainable and recommends the evaluation metric. Every plot is also saved as a CSV so later agents can read it. |
| `domain_research` | Adds knowledge the data does not contain, from the customer or from the web, and proposes features based on it. |
| `preprocess` | Cleans, encodes and imputes the data and splits off a development set. It never creates features. |
| `baseline` | Builds simple rules without a learned model. The best rule sets the floor every model has to beat. |
| `feature_engineering` | The only skill that creates features or decides which ones to keep. It also removes columns that would leak the answer. |
| `training` | Fits one model family, evaluates it on the development set and logs the iteration to MLflow. |
| `diagnosis` | Checks whether the goal is met, names one root cause and proposes a few ranked next steps. It never picks the next skill itself. |
| `end_model` | Chooses the best model per horizon and retrains it on all the training data. |
| `report_generator` | Turns the output of a finished skill into an HTML report. |
| `final_report` | Started by the user. It is the only skill that sees the test set, and it compares the pipeline against AutoML and a naive rule. |

Skills never talk to each other directly. Each one writes its result to a shared state file and to one history file per iteration. Each history file ends with a **handoff block** that holds every value the next skill depends on. A checking script validates that block before the planner continues. Because of this, the agent's context can be cleared after every step, and a run can be stopped, resumed or replayed.

## Results

The three approaches were compared on test data that none of them saw during training. The scores are mean absolute error, in units of bread per day (lower is better).

| Horizon | Naive rule | AutoML (FLAML) | Agentic pipeline |
|---|---|---|---|
| t+1 | 39.8 | 67.3 | **39.3** |
| t+7 | 41.4 | 67.0 | **41.2** |
| t+14 | **42.4** | 69.4 | 44.9 |
| t+30 | 71.1 | 59.1 | **43.0** |
| t+60 | 73.0 | 63.8 | **46.3** |
| t+90 | 85.2 | **61.4** | 64.9 |
| pooled | 58.8 | 64.7 | **46.6** |

Up to two weeks ahead, the naive rule and the pipeline are about equal, because the weekly pattern is very stable. From one month ahead the pipeline loses accuracy much more slowly. This comes from its trend and calendar features, and from a decision it made itself. At most horizons it predicts the ratio to the naive rule instead of the number of units sold.

## Repository layout

| Folder | Contents |
|---|---|
| [`skills/`](skills) | The planner and all the skills, with their reference notes |
| [`contracts/`](contracts) | The JSON input and output contract of each skill |
| [`conventions/`](conventions) | The handoff contract, naming rules and code and plot conventions |
| [`tools/`](tools) | Helper scripts: handoff checker, stop checker, state pruning, report builders and test set creation |
| [`runs/run_20260908T075709Z/`](runs/run_20260908T075709Z) | The final run: code written by the agents, history, state, plots and HTML reports |
| [`automl-runs/`](automl-runs) | The FLAML AutoML baseline packages used for comparison |
| [`holdout_eval/`](holdout_eval) | Scripts for scoring the models on the test set |
| [`mlflow_prototype/`](mlflow_prototype) | A first prototype of MLflow experiment tracking |
| [`graph/`](graph) | Diagrams of the pipeline |
| [`presentation/`](presentation) | The final presentation given at Element61 |

To browse the final run, start from [`reports/index.html`](runs/run_20260908T075709Z/reports/index.html). The comparison on the test set is in [`reports/final_comparison.html`](runs/run_20260908T075709Z/reports/final_comparison.html). Download the HTML files and open them in a browser, because GitHub shows them as source code.

## What is not included

- **Data.** The bakery dataset belongs to an Element61 customer and is not public. All raw data, processed data and CSV outputs (features, predictions and plot data) are left out. The plots and reports are included.
- **Trained models.** The model files are left out, because some are too large for GitHub. They can be rebuilt from the code in each run.

## Tools

Claude Code with agent skills (Claude Sonnet 5 for most runs, Claude Opus 5 for comparison), Python 3.12, pandas, scikit-learn, LightGBM, XGBoost, statsmodels, Optuna, FLAML for the AutoML baseline and MLflow for experiment tracking.
