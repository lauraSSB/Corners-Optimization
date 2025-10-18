# Library imports
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv('unified_corner_analysis_p0_p1_20251017_104945.csv')

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
    'P1_total_n_zone_14'
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

# Prepare features and target
X = df_1[numerical_features + categorical_features]
y = df_1['xg_20s']

# Check the actual distribution
print("xG Distribution Analysis:")
print(f"Mean xG: {df_1['xg_20s'].mean():.4f}")
print(f"Median xG: {df_1['xg_20s'].median():.4f}")
print(f"Min xG: {df_1['xg_20s'].min():.4f}")
print(f"Max xG: {df_1['xg_20s'].max():.4f}")
print(f"Standard Deviation: {df_1['xg_20s'].std():.4f}")

# Check percentiles
percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    value = df_1['xg_20s'].quantile(p/100)
    print(f"{p}th percentile: {value:.4f}")

# Check how many are zero
zero_count = (df_1['xg_20s'] == 0).sum()
total_count = len(df_1)
print(f"Zero xG corners: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)")

'''
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100,
    cat_features=categorical_features
)

# Train the model
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    verbose=100
)

# Make predictions
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("REGRESSION MODEL EVALUATION")
print("=" * 50)
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R² Score: {r2:.4f}")
print(f"Target Mean: {y_test.mean():.4f}")
print(f"Target Std: {y_test.std():.4f}")

# Additional statistical analysis
residuals = y_test - y_pred
print(f"\nResiduals Analysis:")
print(f"Mean Residual: {residuals.mean():.6f}")
print(f"Std of Residuals: {residuals.std():.6f}")
print(f"Max Residual: {residuals.max():.6f}")
print(f"Min Residual: {residuals.min():.6f}")

# Feature Importance
feature_importance = model.get_feature_importance()
feature_names = numerical_features + categorical_features

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n" + "=" * 50)
print("TOP 15 FEATURE IMPORTANCES")
print("=" * 50)
print(importance_df.head(15))

# Plot Feature Importance
plt.figure(figsize=(12, 10))
top_features = importance_df.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances - xG Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Learning Curve
plt.figure(figsize=(10, 6))
if hasattr(model, 'evals_result_'):
    train_metric = list(model.evals_result_['learn'].values())[0]
    val_metric = list(model.evals_result_['validation'].values())[0]
    
    plt.plot(train_metric, label='Training RMSE', alpha=0.7)
    plt.plot(val_metric, label='Validation RMSE', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
'''