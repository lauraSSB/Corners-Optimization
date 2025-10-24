# **Corners Optimization**

Machine Learning model for classifying **high-threat corner kicks** using event data and positional features.  
Built with **CatBoost**, this project identifies tactical and spatial patterns that increase the likelihood of goal-scoring opportunities within 20 seconds after a corner.

## **1. Objective**

To classify each corner as:

    High-Threat = 1   if xG_20s > 0.0435  
    Low-Threat  = 0   otherwise

The 75th percentile of the 20-second xG distribution was selected as the threshold, representing the top quartile of genuinely dangerous corners.

## **2. Dataset and Temporal Split**

Data from **2021/22 to 2024/25 seasons** was used.  
The model learns from historical corner behavior while capturing new tactical patterns. 

---

## **3. Model Configuration and Results**

The model was built with **CatBoostClassifier**, chosen for its ability to handle categorical variables and class imbalance using `auto_class_weights='SqrtBalanced'`.  
The configuration below produced stable convergence and reliable performance without overfitting.

| Parameter | Value | Description |
|------------|--------|-------------|
| depth | 6 | Tree depth (controls model complexity) |
| learning_rate | 0.05 | Step size per iteration |
| l2_leaf_reg | 3.0 | L2 regularization coefficient |
| border_count | 254 | Number of feature split bins |
| random_strength | 1.0 | Randomness applied during tree building |
| bagging_temperature | 1.0 | Controls sampling diversity |

**Evaluation Metrics (Baseline Model)**

| Metric | Low-Threat | High-Threat |
|--------|-------------|-------------|
| **Precision** | 0.88 | 0.59 |
| **Recall** | 0.86 | 0.63 |
| **F1-score** | 0.87 | 0.61 |

**Global metrics:**  
AUC = **0.82**  
PR-AUC = **0.69**  
Weighted F1 = **0.81**

The model achieved strong discriminative ability between high-threat and low-threat corners, with balanced recall and precision across classes.  
While slightly conservative toward the dominant (low-threat) class, the classifier consistently identified dangerous corner patterns with minimal false positives.


## **5. Explainability**

Feature attribution was computed using SHAP values and gain-based importance.  
The most influential features include:
- Number of attackers and defenders in key zones  
- Goalkeeper position and orientation  
- Ball trajectory and corner delivery type  

These features determine the probability of a corner producing a high-threat play.

## 🏁 **How to run**

### 📥 Extract Data
    python Corners-Optimization/modular_code/data_preprocessing_feature_engineering main.py
This script downloads and processes all corner event data from seasons 2021 to 2025, generating the base datasets used for modeling.

### 🧹 Preprocessing for model
    Run Corners-Optimization/modular_code/model/DataPreprocessing.ipynb
  Execute DataPreprocessing.ipynb using the .csv file obtained in the previous step.
  This stage filters valid freeze-frames, cleans redundant columns, and prepares the final dataset for training.

### 🤖 Model 
    python Corners-Optimization/modular_code/model/catboost_high_threat.py \
    --data path/to/final_dataset.csv \
    --xg_col xg_20s \
    --id_cols match_id,event_id \
    --drop_cols "list,of,columns,to,drop" \
    --val_size 0.2

  This script trains the CatBoost classification model, computes SHAP value explanations, and generates model evaluation metrics.

    
