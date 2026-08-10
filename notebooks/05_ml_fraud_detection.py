# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,ML Fraud Detection Model
# Databricks notebook source
# MAGIC %md
# MAGIC # ML Fraud Detection Model Training
# MAGIC 
# MAGIC This notebook trains an XGBoost fraud detection model with:
# MAGIC * Feature engineering from Silver layer
# MAGIC * Model training with hyperparameter tuning
# MAGIC * MLflow experiment tracking
# MAGIC * Unity Catalog model registration
# MAGIC * Model evaluation and performance metrics
# MAGIC * Model deployment for real-time scoring

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# Install required ML libraries
%uv pip install xgboost scikit-learn matplotlib seaborn

# COMMAND ----------

# DBTITLE 1,Restart Python
# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and Configuration
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql.functions import col, lit, current_timestamp

# Configuration
CATALOG = "main"
SCHEMA = "insurance_claims"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_detection_model"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Set MLflow experiment
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/insurance-fraud-detection")

print(f"Training fraud detection model")
print(f"Data source: {CATALOG}.{SCHEMA}.silver_fraud_features")
print(f"Model will be registered to: {MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Load and Prepare Data
# Load feature table from Silver layer
features_df = spark.table(f"{CATALOG}.{SCHEMA}.silver_fraud_features").toPandas()

print(f"Loaded {len(features_df):,} records")
print(f"Features: {features_df.shape[1]} columns")
print(f"\nFraud distribution:")
print(features_df['is_fraud'].value_counts())
print(f"\nFraud rate: {features_df['is_fraud'].mean()*100:.2f}%")

# COMMAND ----------

# DBTITLE 1,Feature Selection and Data Split
# Define features and target
feature_cols = [
    'days_to_file',
    'days_since_policy_start',
    'days_until_policy_end',
    'incident_month',
    'incident_day_of_week',
    'is_weekend_incident',
    'claim_amount',
    'claim_to_limit_ratio',
    'claim_to_premium_ratio',
    'claim_exceeds_limit',
    'claim_near_limit',
    'claim_sequence_num',
    'is_first_claim',
    'total_claims_count',
    'avg_claim_amount',
    'max_claim_amount',
    'prior_fraud_count',
    'provider_count',
    'rapid_filing',
    'delayed_filing',
    'policy_new_incident',
    'policy_expiring_soon',
    'frequent_claimant',
    'high_value_claimant',
    'has_prior_fraud',
    'description_length',
    'description_word_count',
    'has_medical_code'
]

X = features_df[feature_cols]
y = features_df['is_fraud'].astype(int)
claim_ids = features_df['claim_id']

print(f"\nUsing {len(feature_cols)} features for training")
print(f"Feature list: {', '.join(feature_cols[:5])}... (showing first 5)")

# Train/test split (stratified to maintain fraud ratio)
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, claim_ids, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nTrain set: {len(X_train):,} samples ({y_train.mean()*100:.2f}% fraud)")
print(f"Test set:  {len(X_test):,} samples ({y_test.mean()*100:.2f}% fraud)")

# COMMAND ----------

# DBTITLE 1,Train XGBoost Model with MLflow
# Start MLflow run
with mlflow.start_run(run_name="xgboost_fraud_detector") as run:
    
    # Log parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
        'eval_metric': ['logloss', 'auc']
    }
    
    mlflow.log_params(params)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    # Train model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("✓ Model training complete")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Log metrics
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc_roc,
        'avg_precision': avg_precision
    })
    
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc_roc:.4f}")
    print(f"  Avg Prec:  {avg_precision:.4f}")

# COMMAND ----------

# DBTITLE 1,Feature Importance Analysis
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance.head(15).plot(x='feature', y='importance', kind='barh', ax=ax)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances')
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()

# COMMAND ----------

# DBTITLE 1,Confusion Matrix and ROC Curve
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()
    
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]:,}")
    print(f"False Positives: {cm[0,1]:,}")
    print(f"False Negatives: {cm[1,0]:,}")
    print(f"True Positives:  {cm[1,1]:,}")
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close()
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {avg_precision:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, "precision_recall_curve.png")
    plt.close()

# COMMAND ----------

# DBTITLE 1,Register Model to Unity Catalog
    # Create model signature
    signature = infer_signature(X_train, y_pred_proba)
    
    # Log model to MLflow
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_NAME,
        input_example=X_train.head(5)
    )
    
    print(f"\n✓ Model logged to MLflow")
    print(f"\u2713 Model registered to Unity Catalog: {MODEL_NAME}")
    print(f"\nRun ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")

# COMMAND ----------

# DBTITLE 1,Save Predictions to Lakehouse
# Create predictions DataFrame and save to Delta table
predictions_df = pd.DataFrame({
    'claim_id': ids_test.values,
    'actual_fraud': y_test.values,
    'predicted_fraud': y_pred,
    'fraud_probability': y_pred_proba,
    'prediction_correct': (y_test.values == y_pred).astype(int)
})

# Convert to Spark DataFrame and save to Lakehouse
predictions_spark = spark.createDataFrame(predictions_df)
predictions_spark = predictions_spark.withColumn('model_version', lit('v1')) \
    .withColumn('prediction_timestamp', current_timestamp())

predictions_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.ml_fraud_predictions")

print(f"\n✓ Saved {len(predictions_df):,} predictions to {CATALOG}.{SCHEMA}.ml_fraud_predictions")
print("\nSample predictions:")
predictions_spark.orderBy(col('fraud_probability').desc()).show(10)

# COMMAND ----------

# DBTITLE 1,Model Performance Summary
# Create detailed classification report
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

print("\nBusiness Impact Analysis:")
total_test_claims = len(y_test)
actual_fraud = y_test.sum()
detected_fraud = ((y_test == 1) & (y_pred == 1)).sum()
missed_fraud = ((y_test == 1) & (y_pred == 0)).sum()
false_alarms = ((y_test == 0) & (y_pred == 1)).sum()

print(f"  Total Test Claims:     {total_test_claims:,}")
print(f"  Actual Fraud Cases:    {actual_fraud:,}")
print(f"  Detected Fraud:        {detected_fraud:,} ({detected_fraud/actual_fraud*100:.1f}% of actual)")
print(f"  Missed Fraud:          {missed_fraud:,}")
print(f"  False Alarms:          {false_alarms:,}")
print(f"  Detection Rate:        {recall:.1%}")
print(f"  Precision:             {precision:.1%}")

print("\n" + "="*70)
print("Model ready for deployment!")
print(f"Registered model: {MODEL_NAME}")
print(f"Predictions stored in: {CATALOG}.{SCHEMA}.ml_fraud_predictions")
print("\nNext step: Run 06_vector_search_setup notebook")
print("="*70)

# COMMAND ----------

