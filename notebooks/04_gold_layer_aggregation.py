# Databricks notebook source
# DBTITLE 1,Gold Layer - Business Aggregations
# MAGIC %md
# MAGIC # Gold Layer - Business-Level Aggregations
# MAGIC
# MAGIC This notebook creates business-ready aggregated tables for:
# MAGIC * Executive dashboards
# MAGIC * Fraud alert monitoring
# MAGIC * Agent performance tracking
# MAGIC * Claims analytics
# MAGIC
# MAGIC All data persisted to Unity Catalog Delta tables in the Lakehouse.

# COMMAND ----------

# DBTITLE 1,Configuration
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from datetime import datetime

CATALOG = "main"
SCHEMA = "insurance_claims"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Creating Gold layer in: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Load Silver Tables
# Load Silver layer tables
silver_claims = spark.table(f"{CATALOG}.{SCHEMA}.silver_claims_enriched")
silver_features = spark.table(f"{CATALOG}.{SCHEMA}.silver_fraud_features")

print(f"Loaded {silver_claims.count():,} enriched claims")
print(f"Loaded {silver_features.count():,} feature records")

# COMMAND ----------

# DBTITLE 1,Gold Table 1: Claims Summary Dashboard
# Create executive summary table
gold_claims_summary = silver_claims.groupBy(
    'claim_type',
    'insurance_provider',
    'address_state',
    date_trunc('month', col('incident_date')).alias('incident_month')
).agg(
    count('*').alias('total_claims'),
    sum('claim_amount').alias('total_claimed_amount'),
    avg('claim_amount').alias('avg_claim_amount'),
    min('claim_amount').alias('min_claim_amount'),
    max('claim_amount').alias('max_claim_amount'),
    sum(when(col('is_fraud') == True, 1).otherwise(0)).alias('fraud_count'),
    (sum(when(col('is_fraud') == True, 1).otherwise(0)) / count('*') * 100).alias('fraud_rate_pct'),
    sum(when(col('status') == 'Approved', col('claim_amount')).otherwise(0)).alias('approved_amount'),
    sum(when(col('status') == 'Denied', 1).otherwise(0)).alias('denied_count'),
    avg('days_to_file').alias('avg_days_to_file')
)

# Add calculated metrics
gold_claims_summary = gold_claims_summary \
    .withColumn('approval_rate_pct', 
        (col('total_claims') - col('denied_count')) / col('total_claims') * 100) \
    .withColumn('avg_approved_amount',
        col('approved_amount') / when(col('total_claims') > col('denied_count'), 
                                      col('total_claims') - col('denied_count')).otherwise(1)) \
    .withColumn('created_at', current_timestamp())

# Save to Gold layer
gold_claims_summary.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_claims_summary")

print(f"✓ Created gold_claims_summary with {gold_claims_summary.count():,} aggregations")
gold_claims_summary.orderBy(desc('total_claims')).show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,Gold Table 2: Fraud Alerts (High-Risk Claims)
# Create fraud alerts table for high-risk claims requiring investigation
gold_fraud_alerts = silver_features.filter(
    (col('is_fraud') == True) | 
    (col('claim_to_limit_ratio') > 0.85) |
    (col('rapid_filing') == 1) |
    (col('has_prior_fraud') == 1) |
    (col('total_claims_count') > 5)
).join(
    silver_claims.select('claim_id', 'claimant_id', 'policy_id', 'insurance_provider', 
                        'full_name', 'description', 'status'),
    'claim_id',
    'inner'
)

# Calculate composite risk score
gold_fraud_alerts = gold_fraud_alerts \
    .withColumn('risk_score',
        (col('claim_to_limit_ratio') * 0.3 +
         col('rapid_filing') * 0.2 +
         col('has_prior_fraud') * 0.3 +
         when(col('total_claims_count') > 5, 0.2).otherwise(0))
    ) \
    .withColumn('alert_priority',
        when(col('risk_score') > 0.7, 'Critical')
        .when(col('risk_score') > 0.5, 'High')
        .when(col('risk_score') > 0.3, 'Medium')
        .otherwise('Low')
    ) \
    .withColumn('investigation_required',
        when((col('is_fraud') == True) | (col('risk_score') > 0.6), True).otherwise(False)
    ) \
    .withColumn('alert_timestamp', current_timestamp())

# Select key columns for alerts
gold_fraud_alerts = gold_fraud_alerts.select(
    'claim_id',
    'claimant_id',
    'policy_id',
    'full_name',
    'insurance_provider',
    'claim_amount',
    'risk_score',
    'alert_priority',
    'investigation_required',
    'is_fraud',
    'status',
    'description',
    'claim_to_limit_ratio',
    'rapid_filing',
    'has_prior_fraud',
    'total_claims_count',
    'alert_timestamp'
)

# Save to Gold layer
gold_fraud_alerts.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_fraud_alerts")

print(f"✓ Created gold_fraud_alerts with {gold_fraud_alerts.count():,} high-risk claims")
print("\nAlert Priority Breakdown:")
gold_fraud_alerts.groupBy('alert_priority').count().orderBy(desc('count')).show()

# COMMAND ----------

# DBTITLE 1,Gold Table 3: Claimant Risk Profiles
# Create claimant risk profile table
gold_claimant_profiles = silver_claims.groupBy('claimant_id', 'full_name', 'address_state').agg(
    count('*').alias('lifetime_claims'),
    sum('claim_amount').alias('lifetime_claimed'),
    avg('claim_amount').alias('avg_claim_size'),
    max('claim_amount').alias('max_claim_size'),
    sum(when(col('is_fraud') == True, 1).otherwise(0)).alias('fraud_claims_count'),
    min('incident_date').alias('first_claim_date'),
    max('incident_date').alias('last_claim_date'),
    countDistinct('insurance_provider').alias('unique_providers'),
    countDistinct('claim_type').alias('unique_claim_types'),
    avg('days_to_file').alias('avg_filing_delay')
)

# Calculate risk indicators
gold_claimant_profiles = gold_claimant_profiles \
    .withColumn('fraud_rate',
        col('fraud_claims_count') / col('lifetime_claims')
    ) \
    .withColumn('days_active',
        datediff(col('last_claim_date'), col('first_claim_date'))
    ) \
    .withColumn('claims_per_year',
        col('lifetime_claims') / when(col('days_active') > 0, col('days_active') / 365).otherwise(1)
    ) \
    .withColumn('risk_tier',
        when(col('fraud_rate') > 0, 'High Risk - Known Fraud')
        .when(col('lifetime_claims') > 5, 'High Risk - Frequent')
        .when(col('claims_per_year') > 2, 'Medium Risk - Active')
        .otherwise('Low Risk')
    ) \
    .withColumn('profile_updated_at', current_timestamp())

# Save to Gold layer
gold_claimant_profiles.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_claimant_profiles")

print(f"✓ Created gold_claimant_profiles with {gold_claimant_profiles.count():,} profiles")
print("\nRisk Tier Distribution:")
gold_claimant_profiles.groupBy('risk_tier').count().orderBy(desc('count')).show()

# COMMAND ----------

# DBTITLE 1,Gold Table 4: Provider Performance Metrics
# Create insurance provider performance table
gold_provider_metrics = silver_claims.groupBy(
    'insurance_provider',
    date_trunc('quarter', col('incident_date')).alias('quarter')
).agg(
    count('*').alias('total_claims'),
    sum('claim_amount').alias('total_exposure'),
    avg('claim_amount').alias('avg_claim_size'),
    sum(when(col('is_fraud') == True, col('claim_amount')).otherwise(0)).alias('fraud_losses'),
    sum(when(col('is_fraud') == True, 1).otherwise(0)).alias('fraud_count'),
    sum(when(col('status') == 'Approved', 1).otherwise(0)).alias('approved_count'),
    sum(when(col('status') == 'Denied', 1).otherwise(0)).alias('denied_count'),
    avg('days_to_file').alias('avg_processing_time'),
    countDistinct('claimant_id').alias('unique_claimants')
)

# Calculate performance metrics
gold_provider_metrics = gold_provider_metrics \
    .withColumn('fraud_rate_pct',
        col('fraud_count') / col('total_claims') * 100
    ) \
    .withColumn('fraud_loss_ratio',
        col('fraud_losses') / col('total_exposure')
    ) \
    .withColumn('approval_rate_pct',
        col('approved_count') / col('total_claims') * 100
    ) \
    .withColumn('claims_per_claimant',
        col('total_claims') / col('unique_claimants')
    ) \
    .withColumn('metrics_date', current_timestamp())

# Save to Gold layer
gold_provider_metrics.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_provider_metrics")

print(f"✓ Created gold_provider_metrics with {gold_provider_metrics.count():,} records")
gold_provider_metrics.orderBy(desc('total_exposure')).show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,Gold Table 5: Time Series Analytics
# Create time series table for trend analysis
gold_time_series = silver_claims.withColumn(
    'incident_week', date_trunc('week', col('incident_date'))
).groupBy('incident_week', 'claim_type').agg(
    count('*').alias('claims_count'),
    sum('claim_amount').alias('total_amount'),
    avg('claim_amount').alias('avg_amount'),
    sum(when(col('is_fraud') == True, 1).otherwise(0)).alias('fraud_count'),
    sum(when(col('status') == 'Approved', 1).otherwise(0)).alias('approved_count')
)

# Add moving averages for trend detection
window_spec = Window.partitionBy('claim_type').orderBy('incident_week').rowsBetween(-3, 0)

gold_time_series = gold_time_series \
    .withColumn('claims_4week_avg', avg('claims_count').over(window_spec)) \
    .withColumn('amount_4week_avg', avg('total_amount').over(window_spec)) \
    .withColumn('fraud_rate_pct', 
        col('fraud_count') / col('claims_count') * 100
    ) \
    .withColumn('created_at', current_timestamp())

# Save to Gold layer
gold_time_series.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_time_series")

print(f"✓ Created gold_time_series with {gold_time_series.count():,} time periods")
gold_time_series.orderBy(desc('incident_week')).show(10)

# COMMAND ----------

# DBTITLE 1,Summary and Data Quality
# Gold layer summary
print("\n" + "="*60)
print("GOLD LAYER CREATION COMPLETE")
print("="*60)

print(f"\nGold tables created in {CATALOG}.{SCHEMA}:")
print("\n1. gold_claims_summary")
print("   Purpose: Executive dashboard metrics by type, provider, state, month")
print(f"   Records: {spark.table(f'{CATALOG}.{SCHEMA}.gold_claims_summary').count():,}")

print("\n2. gold_fraud_alerts")
print("   Purpose: High-risk claims requiring investigation")
print(f"   Records: {spark.table(f'{CATALOG}.{SCHEMA}.gold_fraud_alerts').count():,}")

print("\n3. gold_claimant_profiles")
print("   Purpose: Claimant risk profiles and behavioral patterns")
print(f"   Records: {spark.table(f'{CATALOG}.{SCHEMA}.gold_claimant_profiles').count():,}")

print("\n4. gold_provider_metrics")
print("   Purpose: Insurance provider performance tracking")
print(f"   Records: {spark.table(f'{CATALOG}.{SCHEMA}.gold_provider_metrics').count():,}")

print("\n5. gold_time_series")
print("   Purpose: Weekly time series for trend analysis")
print(f"   Records: {spark.table(f'{CATALOG}.{SCHEMA}.gold_time_series').count():,}")

print("\n" + "="*60)
print("All Gold tables stored as Delta tables in Unity Catalog")
print("Ready for:")
print("  • Databricks SQL dashboards")
print("  • BI tool integration")
print("  • Agent decision systems")
print("  • Real-time monitoring")
print("\nNext step: Run 05_ml_fraud_detection notebook")
print("="*60)

# COMMAND ----------

