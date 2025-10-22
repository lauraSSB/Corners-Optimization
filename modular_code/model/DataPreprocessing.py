# Library imports
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Loading data
df2025 = pd.read_csv('/Users/roudarreza/Documents/isac_2025/modular_code_3/data_preprocessing_feature_engineering/liga_mx_2024_2025.csv')
df2024 = pd.read_csv('/Users/roudarreza/Documents/isac_2025/modular_code_3/data_preprocessing_feature_engineering/liga_mx_2023_2024.csv')
df2023 = pd.read_csv('/Users/roudarreza/Documents/isac_2025/modular_code_3/data_preprocessing_feature_engineering/liga_mx_2022_2023.csv')
df2022 = pd.read_csv('/Users/roudarreza/Documents/isac_2025/modular_code_3/data_preprocessing_feature_engineering/liga_mx_2022_2023.csv')

# Creating a copy of the data
_df2025 = df2025.copy()
_df2024 = df2024.copy()
_df2023 = df2023.copy()
_df2022 = df2023.copy()

df2325 = pd.concat([_df2025, _df2024, _df2023, _df2022])

# Excluding IDs, timestamps, and helper columns from preprocessing.
drop_cols = [
    'match_id', 'event_id', 'minute', 'second', 'period', 'pass_outcome', 'team', 'player', 'pass_type', 'play_pattern',
    'recipient', 'P0_index_x', 'P0_index_y','corner_execution_time_raw', 'match_date', 
    'home_team', 'away_team', 'season', 'P1_event_id', 'P1_index', 'P1_timestamp', 
    'corner_execution_time_raw', 'zone_1_name', 'P0_total_n_zone_1', 'zone_3_name',
    'P0_total_n_zone_3', 'zone_4_name', 'P0_total_n_zone_4', 'zone_5_name', 'P0_total_n_zone_5',
    'zone_6_name', 'P0_total_n_zone_6', 'zone_7_name', 'P0_total_n_zone_7',
    'zone_8_name', 'P0_total_n_zone_8', 'zone_9_name', 'P0_total_n_zone_9',
    'zone_10_name', 'P0_total_n_zone_10', 'zone_11_name', 'P0_total_n_zone_11',
    'zone_12_name', 'P0_total_n_zone_12', 'zone_13_name', 'P0_total_n_zone_13',
    'zone_14_name', 'P0_total_n_zone_14', 'P1_type', 'P1_team', 'P1_coordinates_normalized', 'P1_total_n_zone_1',
    'P1_total_n_zone_3', 'P1_total_n_zone_4', 'P1_total_n_zone_5',
    'P1_total_n_zone_6', 'P1_total_n_zone_7', 'P1_total_n_zone_8', 'P1_total_n_zone_9',
    'P1_total_n_zone_10', 'P1_total_n_zone_11', 'P1_total_n_zone_12', 'P1_total_n_zone_13',
    'P1_total_n_zone_14', 'goal_20s', 'goal_20s_def', 'xg_20s_def', 'P0_index'
]

df_1 = df2325.drop(columns=drop_cols)

# Excluding target-related columns
exclude_cols = [
    'xg_20s'
]

# Select numerical and categorical features
feature_columns = [col for col in df_1.columns if col not in exclude_cols]

# Separate numerical and categorical features
numerical_features = []
categorical_features = []

for col in feature_columns:
    if df_1[col].dtype in ['int64', 'float64']:
        numerical_features.append(col)
    else:
        categorical_features.append(col)

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

nan_df = (
    df_1.isna()
        .sum()
        .to_frame('n_missing')
        .assign(pct_missing=lambda x: (x['n_missing'] / len(df_1)) * 100)
        .sort_values('n_missing', ascending=False)
        .reset_index()
        .rename(columns={'index': 'column'})
)

print(nan_df.head())

df_1 = df_1.dropna(subset=[
    'P0_GK_y', 'P0_GK_x', 'P1_GK_x', 'P1_GK_y', 
    'P0_n_att_zone_14', 'P1_n_def_zone_14'
])

nan_cleaned_df = (
    df_1.isna()
        .sum()
        .to_frame('n_missing')
        .assign(pct_missing=lambda x: (x['n_missing'] / len(df_1)) * 100)
        .sort_values('n_missing', ascending=False)
        .reset_index()
        .rename(columns={'index': 'column'})
)

print(nan_cleaned_df.head())
print(df_1.shape)

# Check the actual xG distribution
print("xG Distribution Analysis:")
print(f"Mean xG: {df_1['xg_20s'].mean():.4f}")
print(f"Median xG: {df_1['xg_20s'].median():.4f}")
print(f"Min xG: {df_1['xg_20s'].min():.4f}")
print(f"Max xG: {df_1['xg_20s'].max():.4f}")
print(f"Standard Deviation: {df_1['xg_20s'].std():.4f}")

# Check xG percentiles
percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    value = df_1['xg_20s'].quantile(p/100)
    print(f"{p}th percentile: {value:.4f}")

    # Check how many are shots have xG equalt to zero
zero_count = (df_1['xg_20s'] == 0).sum()
total_count = len(df_1)
print(f"Zero xG corners: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)")

df_1.to_csv('liga_mx_2021_2025_preprocessed.csv', index=False)