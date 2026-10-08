"""Update state/current.json after iteration 004 baseline."""
import json
from pathlib import Path

RUN_DIR = Path('C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/runs/run_20260908T075709Z')

with open(RUN_DIR / 'state' / 'current.json') as f:
    state = json.load(f)

with open(RUN_DIR / 'history' / 'iteration_004_baseline.json') as f:
    h4 = json.load(f)

hf  = h4['handoff']
bf  = hf['parameters']['baseline_floor']['value']
npf = hf['parameters']['naive_predictor_features']['value']
pf  = hf['parameters']['protected_features']['value']

# Update current fields
state['current']['task_type'] = 'time_series_forecasting'
state['current']['baseline_floor'] = bf
state['current']['naive_predictor_features'] = {
    'established_at_iteration': 4,
    'selection_basis': 'train_rows_only (fold 3)',
    **npf
}
state['current']['protected_features'] = pf
state['current']['ts_approaches_tried'] = []

# Append history entry
state['history'].append({
    'iteration': 4,
    'node': 'baseline',
    'timestamp': '2026-09-08',
    'status': 'complete',
    'handoff_check': (
        'pass (--iteration-only); pre-existing BLOCK from iteration_003 context_clearable=false '
        'is an acknowledged signal, not a defect in iteration_004'
    ),
    'summary': (
        'Part A: scored 6 trivial rules on 3 folds of validation rows. '
        'Floor per horizon (fold 3 primary, h90 pooled): '
        'h1=38.92 (seasonal_naive), h7=38.92 (last_value), h14=40.18 (last_value), '
        'h30=40.50 (train_mean -- fold3 val mean close to train mean, level_gap=-3.7%), '
        'h60=40.50 (train_mean), h90=43.02 (seasonal_median_52w, pooled 3 folds, n=257). '
        'Part B: selected 3 naive predictor specs per horizon (18 total protected features). '
        'smedian_w13/w26 dominate short-horizon train search.'
    ),
    'why_this_node': (
        'First Training call must be baseline mode to establish baseline_floor and '
        'naive_predictor_features before any FE call. Hard ordering constraint per planner.md.'
    ),
    'candidates_considered': 'None -- baseline is the mandatory first Training call per the ordering rule.',
    'full_output_path': 'history/iteration_004_baseline.json',
    'report_html': None,
    'key_outputs': {
        'baseline_floor_per_h': {
            str(h): {
                'winner': bf['per_horizon'][str(h)]['winner'],
                'mae': bf['per_horizon'][str(h)]['winner_mae']
            }
            for h in [1, 7, 14, 30, 60, 90]
        },
        'pooled_floor_mae': bf['pooled']['mae'],
        'n_protected_features': len(pf),
        'artifacts': '2 plots + 2 CSVs in artifacts/training/',
    }
})

# Next iteration contract: feature_engineering
state['current']['next_iteration_contract'] = {
    'node': 'feature_engineering',
    'iteration': 5,
    'why_this_node': (
        'Baseline floor and naive_predictor_features established (iteration 004). '
        'Feature Engineering is next: materialise 18 naive predictor columns as protected features, '
        'build all calendar/domain/lag features from domain research, produce long-format FE CSV '
        'per horizon. No other node can run next: Training needs a feature set; Diagnosis needs a model.'
    ),
    'read_this_first': (
        '18 naive predictor columns (3 per horizon) must be built EXACTLY per formula in '
        'handoff.parameters.naive_predictor_features. Horizon-scoped: naive_h7_* only valid for h=7 '
        'model. train_mean winner for h30/h60 floor -- note naive_h30_train_mean and '
        'naive_h60_train_mean will be constant within each FE build window (fold 3 train mean=317.68). '
        'had_reduction must be dropped (99.9% constant). '
        'Leakage cols excluded: pistolet_quantity, total_quantity, total_revenue, bread_revenue, '
        'num_orders, payment_mode_1/2/4.'
    ),
    'required_state_fields': [
        'baseline_floor', 'naive_predictor_features', 'protected_features',
        'forecast_horizons', 'target_column', 'eda_summary'
    ],
    'context': {
        'processed_csv': 'data/iteration_003_preprocess.csv',
        'processed_shape': [2897, 35],
        'target_column': 'bread_quantity',
        'forecast_horizons': [1, 7, 14, 30, 60, 90],
        'naive_predictor_features': npf,
        'protected_features': pf,
        'baseline_floor_per_horizon': {
            str(h): {
                'winner': bf['per_horizon'][str(h)]['winner'],
                'mae': bf['per_horizon'][str(h)]['winner_mae']
            }
            for h in [1, 7, 14, 30, 60, 90]
        },
        'split_column': '_split',
        'fold_columns': ['_fold_1', '_fold_2', '_fold_3'],
        'fold3_split': {
            'train_end': '2020-06-30', 'gap_end': '2020-09-27',
            'val_start': '2020-09-28', 'val_end': '2020-12-31',
            'train_rows': 2713, 'gap_rows': 89, 'val_rows': 95,
            'scoreable_val_rows': {'h1': 94, 'h7': 88, 'h14': 81, 'h30': 65, 'h60': 35, 'h90': 5}
        },
        'leakage_risk_cols_do_not_use': [
            'pistolet_quantity', 'total_quantity', 'total_revenue', 'bread_revenue',
            'num_orders', 'payment_mode_1', 'payment_mode_2', 'payment_mode_4'
        ],
        'near_constant_drop': ['had_reduction'],
        'weather_columns': [
            'wx_Maximum_Temperature', 'wx_Minimum_Temperature', 'wx_Temperature',
            'wx_Precipitation', 'wx_Snow_depth', 'wx_Wind_Speed', 'wx_Wind_Gust',
            'wx_Cloud_Cover', 'wx_Relative_Humidity'
        ],
        'competitor_columns': [
            'comp_n_competitors_closed', 'comp_any_closed',
            'comp_closure_density', 'comp_data_available'
        ],
        'regime_change_date': '2020-06-01',
        'h90_instruction': 'Evaluate h=90 model candidates using all 3 folds -- fold 3 has only 5 scoreable val rows',
        'seasonal_period_P': 7,
        'seasonal_window_S': 52,
        'domain_research_features': [
            'is_official_public_holiday (11 Belgian holidays, floating via Easter formula -- NOT hardcoded)',
            'days_until_next_holiday (capped 30), days_since_last_holiday (capped 14)',
            'is_school_holiday_flanders (Easter-anchored Krokus/Paas; fixed Jul1-Aug31 Zomervakantie)',
            'is_zomervakantie (Jul 1 - Aug 31)',
            'is_gentse_feesten (year-specific dates from domain_research; 2020=0)',
            'year_trend = (date - 2010-11-16).days / 365.25',
            'regime_x_monday (date >= 2020-06-01 AND dow==0) -- critical Monday training signal',
            'is_2014_sparse_period (year==2014)',
            'is_hot_day (wx_Temperature > 25), is_heavy_rain (wx_Precipitation > 5)',
            'DOW-aligned lag features per horizon h: lag = k*7 where k=ceil(h/7)',
            'is_covid_lockdown_period (2020-03-18 to 2020-05-03)',
            'is_sinterklaas_period (Dec 1-6), is_easter_period (Good Fri to Easter Mon)',
        ],
        'preprocess_history_path': 'history/iteration_003_preprocess.json',
        'domain_research_history_path': 'history/iteration_002_domain_research.json',
    }
}

with open(RUN_DIR / 'state' / 'current.json', 'w') as f:
    json.dump(state, f, indent=2, default=str)

print('state/current.json updated successfully')
print('History entries:', [(h['iteration'], h['node']) for h in state['history']])
print('Next iteration contract:', state['current']['next_iteration_contract']['node'],
      'iteration', state['current']['next_iteration_contract']['iteration'])
print('baseline_floor pooled MAE:', state['current']['baseline_floor']['pooled']['mae'])
print('Protected features:', len(state['current']['protected_features']))
