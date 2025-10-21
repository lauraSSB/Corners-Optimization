# Library imports
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
import matplotlib.pyplot as plt
from DataPreprocessing import data_preprocessing_model

# Loading data
df, numerical_features, categorical_features = data_preprocessing_model("liga_mx_2023_2025.csv")

df_1 = df.copy()

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

df_1['has_xg'] = (df_1['xg_20s'] > 0.05).astype(int)
print("Has xG distribution:")
print(df_1['has_xg'].value_counts())
print(f"Proportion with xG > 0: {df_1['has_xg'].mean():.3f}")

# Stage 2: Predict xG value only for corners that have xG > 0
df_positive_xg = df_1[df_1['xg_20s'] > 0.05].copy()
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
    cat_features=categorical_features,
    auto_class_weights='Balanced'
)


print("\nTraining Stage 1 Model...")
model_stage1.fit(
    X_train1, y_train1,
    eval_set=(X_test1, y_test1),
    early_stopping_rounds=50,
    verbose=100
)
print("---------------------------------------------------------------")
print(X_train1.columns)

model_stage1.save_model("catboost_corners_model_clasification.cbm")

# Evaluate Stage 1
y_pred1 = model_stage1.predict(X_test1)
y_pred_proba1 = model_stage1.predict_proba(X_test1)[:, 1]


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

prec, rec, thresh = precision_recall_curve(y_test1, y_pred_proba1)

plt.plot(thresh, prec[:-1], label='Precision')
plt.plot(thresh, rec[:-1], label='Recall')
plt.xlabel("Threshold")
plt.legend()
plt.title("Precision–Recall Tradeoff")
plt.show()


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