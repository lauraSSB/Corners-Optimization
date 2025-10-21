# Library imports
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv('liga_mx_2024_2025.csv')

df_1 = df.copy()

# Identify feature columns (excluding IDs, timestamps, and target-related columns)
drop_cols = [
    'match_id', 'minute', 'second', 'period', 'event_id', 'team', 'player', 'pass_type', 'play_pattern',
    'recipient', 'P0_index_x', 'P0_index_y','corner_execution_time_raw', 'match_date', 
    'home_team', 'away_team', 'season', 'P1_event_id', 'P1_index', 'P1_timestamp', 
    'corner_execution_time_raw', 'zone_1_name', 'P0_total_n_zone_1', 'zone_3_name',
    'P0_total_n_zone_3', 'zone_4_name', 'P0_total_n_zone_4', 'zone_5_name', 'P0_total_n_zone_5',
    'zone_6_name', 'P0_total_n_zone_6', 'zone_7_name', 'P0_total_n_zone_7',
    'zone_8_name', 'P0_total_n_zone_8', 'zone_9_name', 'P0_total_n_zone_9',
    'zone_10_name', 'P0_total_n_zone_10', 'zone_11_name', 'P0_total_n_zone_11',
    'zone_12_name', 'P0_total_n_zone_12', 'zone_13_name', 'P0_total_n_zone_13',
    'zone_14_name', 'P0_total_n_zone_14', 'pass_outcome', 'P1_type', 'P1_total_n_zone_1',
    'P1_total_n_zone_3', 'P1_total_n_zone_4', 'P1_total_n_zone_5',
    'P1_total_n_zone_6', 'P1_total_n_zone_7', 'P1_total_n_zone_8', 'P1_total_n_zone_9',
    'P1_total_n_zone_10', 'P1_total_n_zone_11', 'P1_total_n_zone_12', 'P1_total_n_zone_13',
    'P1_total_n_zone_14', 'location_x',	'location_y',
]

df_1 = df_1.drop(columns=drop_cols)

exclude_cols = [
    'goal_20s', 'goal_20s_def', 'xg_20s', 'xg_20s_def',  # target-related
]

# Select numerical and categorical features
feature_columns = [col for col in df_1.columns if col not in exclude_cols]

# Separate numerical and categorical features
numerical_features = []
categorical_features = []

for col in feature_columns:
    if df[col].dtype in ['int64', 'float64']:
        numerical_features.append(col)
    else:
        categorical_features.append(col)

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

df_1 = df_1.dropna(subset=['P0_n_defenders_in_18yd_box', 'P1_n_defenders_in_18yd_box'])

df_1['P1_GK_x'] = df_1['P1_GK_x'].fillna(-1)
df_1['P1_GK_y'] = df_1['P1_GK_y'].fillna(-1)

# df_1['pass_outcome'] = df_1['pass_outcome'].fillna('Undefined')

df_1['has_xg'] = (df_1['xg_20s'] > 0).astype(int)
print("Has xG distribution:")
print(df_1['has_xg'].value_counts())
print(f"Proportion with xG > 0: {df_1['has_xg'].mean():.3f}")

# Stage 2: Predict xG value only for corners that have xG > 0
df_positive_xg = df_1[df_1['xg_20s'] > 0].copy()
print(f"\nPositive xG corners: {len(df_positive_xg)}")
print(f"Mean xG when > 0: {df_positive_xg['xg_20s'].mean():.4f}")
print(f"Median xG when > 0: {df_positive_xg['xg_20s'].median():.4f}")
print(f"Std xG when > 0: {df_positive_xg['xg_20s'].std():.4f}")

# Check distribution of positive xG values
print(f"\nPositive xG distribution percentiles:")
percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    value = df_positive_xg['xg_20s'].quantile(p/100)
    print(f"  {p}th percentile: {value:.4f}")

# Prepare features for both stages
X = df_1[numerical_features + categorical_features]
X_positive = df_positive_xg[numerical_features + categorical_features]

# Split data for Stage 1
y_stage1 = df_1['has_xg']
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y_stage1, test_size=0.2, random_state=42, stratify=y_stage1
)

print(f"\nStage 1 - Training samples: {X_train1.shape[0]}")
print(f"Stage 1 - Test samples: {X_test1.shape[0]}")

# Train Stage 1 Model (Classification)
model_stage1 = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    random_seed=42,
    verbose=100,
    cat_features=categorical_features
)

print("\nTraining Stage 1 Model...")
model_stage1.fit(
    X_train1, y_train1,
    eval_set=(X_test1, y_test1),
    early_stopping_rounds=50,
    verbose=100
)

# Evaluate Stage 1
y_pred1 = model_stage1.predict(X_test1)
y_pred_proba1 = model_stage1.predict_proba(X_test1)[:, 1]

print("\n" + "=" * 50)
print("STAGE 1 RESULTS: Predicting xG > 0")
print("=" * 50)
print(classification_report(y_test1, y_pred1))
print(f"ROC-AUC: {roc_auc_score(y_test1, y_pred_proba1):.4f}")

# Feature importance for Stage 1
feature_importance1 = model_stage1.get_feature_importance()
importance_df1 = pd.DataFrame({
    'feature': numerical_features + categorical_features,
    'importance': feature_importance1
}).sort_values('importance', ascending=False)

print("\nTop 15 Features for Predicting xG > 0:")
print(importance_df1.head(15))

# Stage 2: Predict xG value for corners that have xG > 0
print("\n" + "=" * 60)
print("STAGE 2: PREDICTING xG VALUE FOR CORNERS WITH xG > 0")
print("=" * 60)

y_stage2 = df_positive_xg['xg_20s']
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_positive, y_stage2, test_size=0.2, random_state=42
)

print(f"Stage 2 - Training samples: {X_train2.shape[0]}")
print(f"Stage 2 - Test samples: {X_test2.shape[0]}")
print(f"Stage 2 - Target range: {y_stage2.min():.4f} to {y_stage2.max():.4f}")

# Train Stage 2 Model (Regression)
model_stage2 = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100,
    cat_features=categorical_features
)

print("\nTraining Stage 2 Model...")
model_stage2.fit(
    X_train2, y_train2,
    eval_set=(X_test2, y_test2),
    early_stopping_rounds=50,
    verbose=100
)

# Evaluate Stage 2
y_pred2 = model_stage2.predict(X_test2)

mse_stage2 = mean_squared_error(y_test2, y_pred2)
rmse_stage2 = np.sqrt(mse_stage2)
mae_stage2 = mean_absolute_error(y_test2, y_pred2)
r2_stage2 = 1 - (mse_stage2 / np.var(y_test2))

print("\n" + "=" * 50)
print("STAGE 2 RESULTS: Predicting xG Value")
print("=" * 50)
print(f"Mean Squared Error (MSE): {mse_stage2:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse_stage2:.6f}")
print(f"Mean Absolute Error (MAE): {mae_stage2:.6f}")
print(f"R² Score: {r2_stage2:.4f}")
print(f"Target Mean: {y_test2.mean():.4f}")
print(f"Target Std: {y_test2.std():.4f}")

# Feature importance for Stage 2
feature_importance2 = model_stage2.get_feature_importance()
importance_df2 = pd.DataFrame({
    'feature': numerical_features + categorical_features,
    'importance': feature_importance2
}).sort_values('importance', ascending=False)

print("\nTop 15 Features for Predicting xG Value (when > 0):")
print(importance_df2.head(15))