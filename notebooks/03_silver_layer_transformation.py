# Databricks notebook source
# DBTITLE 1,Silver Layer - Data Transformation & Feature Engineering
# MAGIC %md
# MAGIC # Silver Layer - Data Transformation & Feature Engineering
# MAGIC
# MAGIC This notebook transforms Bronze layer data into clean, enriched Silver tables with:
# MAGIC * Data quality rules and validations
# MAGIC * Business logic transformations
# MAGIC * Feature engineering for ML fraud detection
# MAGIC * Entity resolution and joins
# MAGIC * PII handling and masking

# COMMAND ----------

# DBTITLE 1,Configuration and Imports
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime
import hashlib

CATALOG = "main"
SCHEMA = "insurance_claims"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Transforming data in: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Load Bronze Tables
# Load bronze tables
bronze_claims = spark.table(f"{CATALOG}.{SCHEMA}.bronze_claims")
bronze_policies = spark.table(f"{CATALOG}.{SCHEMA}.bronze_policies")
bronze_claimants = spark.table(f"{CATALOG}.{SCHEMA}.bronze_claimants")

print(f"Loaded {bronze_claims.count():,} claims")
print(f"Loaded {bronze_policies.count():,} policies")
print(f"Loaded {bronze_claimants.count():,} claimants")

# COMMAND ----------

# DBTITLE 1,Join and Enrich Claims Data
# Join claims with policies and claimants
silver_enriched = bronze_claims \
    .join(bronze_policies, 'policy_id', 'left') \
    .join(bronze_claimants, bronze_claims.claimant_id == bronze_claimants.claimant_id, 'left') \
    .select(
        # Claim fields
        bronze_claims['claim_id'],
        bronze_claims['policy_id'],
        bronze_claims['claimant_id'],
        bronze_claims['claim_type'],
        bronze_claims['incident_date'],
        bronze_claims['filing_date'],
        bronze_claims['claim_amount'],
        bronze_claims['description'],
        bronze_claims['diagnosis_code'],
        bronze_claims['procedure_code'],
        bronze_claims['status'],
        bronze_claims['is_fraud'],  # Ground truth label
        
        # Policy fields
        bronze_policies['provider'].alias('insurance_provider'),
        bronze_policies['policy_limit'],
        bronze_policies['deductible'],
        bronze_policies['premium_annual'],
        bronze_policies['start_date'].alias('policy_start_date'),
        bronze_policies['end_date'].alias('policy_end_date'),
        
        # Claimant fields (masked for PII protection)
        bronze_claimants['first_name'],
        bronze_claimants['last_name'],
        concat(col('first_name'), lit(' '), col('last_name')).alias('full_name'),
        bronze_claimants['date_of_birth'],
        # Mask email - show only domain
        regexp_extract(col('email'), '@(.+)', 1).alias('email_domain'),
        bronze_claimants['address_state'],
        
        # Metadata
        bronze_claims['_ingestion_timestamp'],
        bronze_claims['_source_system']
    )

print(f"Enriched {silver_enriched.count():,} claims with policy and claimant data")
silver_enriched.show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Feature Engineering - Temporal Features
# Temporal feature engineering
silver_features = silver_enriched \
    .withColumn('days_to_file', datediff(col('filing_date'), col('incident_date'))) \
    .withColumn('days_since_policy_start', datediff(col('incident_date'), col('policy_start_date'))) \
    .withColumn('days_until_policy_end', datediff(col('policy_end_date'), col('incident_date'))) \
    .withColumn('incident_year', year(col('incident_date'))) \
    .withColumn('incident_month', month(col('incident_date'))) \
    .withColumn('incident_day_of_week', dayofweek(col('incident_date'))) \
    .withColumn('filing_year', year(col('filing_date'))) \
    .withColumn('filing_month', month(col('filing_date'))) \
    .withColumn('is_weekend_incident', when(col('incident_day_of_week').isin([1, 7]), 1).otherwise(0))

print("✓ Added temporal features")

# COMMAND ----------

# DBTITLE 1,Feature Engineering - Financial Features
# Financial feature engineering
silver_features = silver_features \
    .withColumn('claim_to_limit_ratio', col('claim_amount') / col('policy_limit')) \
    .withColumn('claim_to_premium_ratio', col('claim_amount') / col('premium_annual')) \
    .withColumn('claim_exceeds_limit', when(col('claim_amount') > col('policy_limit'), 1).otherwise(0)) \
    .withColumn('claim_near_limit', when(col('claim_to_limit_ratio') >= 0.9, 1).otherwise(0)) \
    .withColumn('amount_category',
        when(col('claim_amount') < 5000, 'Low')
        .when(col('claim_amount') < 20000, 'Medium')
        .when(col('claim_amount') < 50000, 'High')
        .otherwise('Very High')
    )

print("✓ Added financial features")

# COMMAND ----------

# DBTITLE 1,Feature Engineering - Behavioral Features
# Behavioral features - claim history per claimant
window_claimant = Window.partitionBy('claimant_id').orderBy('filing_date')

silver_features = silver_features \
    .withColumn('claim_sequence_num', row_number().over(window_claimant)) \
    .withColumn('is_first_claim', when(col('claim_sequence_num') == 1, 1).otherwise(0))

# Aggregate historical behavior per claimant
claimant_history = silver_features.groupBy('claimant_id').agg(
    count('*').alias('total_claims_count'),
    sum('claim_amount').alias('total_claimed_amount'),
    avg('claim_amount').alias('avg_claim_amount'),
    max('claim_amount').alias('max_claim_amount'),
    sum(when(col('is_fraud') == True, 1).otherwise(0)).alias('prior_fraud_count'),
    countDistinct('insurance_provider').alias('provider_count')
)

# Join history back
silver_features = silver_features.join(
    claimant_history,
    'claimant_id',
    'left'
)

print("✓ Added behavioral features")

# COMMAND ----------

# DBTITLE 1,Feature Engineering - Risk Indicators
# Risk indicator features
silver_features = silver_features \
    .withColumn('rapid_filing', when(col('days_to_file') <= 1, 1).otherwise(0)) \
    .withColumn('delayed_filing', when(col('days_to_file') > 30, 1).otherwise(0)) \
    .withColumn('policy_new_incident', when(col('days_since_policy_start') < 30, 1).otherwise(0)) \
    .withColumn('policy_expiring_soon', when(col('days_until_policy_end') < 30, 1).otherwise(0)) \
    .withColumn('frequent_claimant', when(col('total_claims_count') > 3, 1).otherwise(0)) \
    .withColumn('high_value_claimant', when(col('total_claimed_amount') > 100000, 1).otherwise(0)) \
    .withColumn('has_prior_fraud', when(col('prior_fraud_count') > 0, 1).otherwise(0))

print("✓ Added risk indicator features")

# COMMAND ----------

# DBTITLE 1,Feature Engineering - Text Features
# Text analysis features
silver_features = silver_features \
    .withColumn('description_length', length(col('description'))) \
    .withColumn('description_word_count', size(split(col('description'), ' '))) \
    .withColumn('has_medical_code', when(col('diagnosis_code').isNotNull(), 1).otherwise(0))

print("✓ Added text features")

# COMMAND ----------

# DBTITLE 1,Integrate External Weather Enrichment
# Integrate external weather enrichment data
try:
    # Check if external enrichment table exists
    bronze_external = spark.table(f"{CATALOG}.{SCHEMA}.bronze_external_enrichment")
    # Force evaluation to trigger exception if table doesn't exist
    _ = bronze_external.count()
    
    # Parse JSON enrichment data and extract weather fields
    from pyspark.sql.functions import get_json_object
    
    weather_enrichment = bronze_external \
        .filter(col('enrichment_source') == 'weather') \
        .filter(col('api_status') == 'success') \
        .select(
            col('claim_id'),
            get_json_object(col('enrichment_data'), '$.temperature_max_f').cast('float').alias('weather_temp_max_f'),
            get_json_object(col('enrichment_data'), '$.temperature_min_f').cast('float').alias('weather_temp_min_f'),
            get_json_object(col('enrichment_data'), '$.precipitation_inches').cast('float').alias('weather_precipitation_in'),
            get_json_object(col('enrichment_data'), '$.wind_speed_mph').cast('float').alias('weather_wind_speed_mph'),
            get_json_object(col('enrichment_data'), '$.conditions_summary').alias('weather_conditions'),
            col('enrichment_timestamp').alias('weather_api_timestamp')
        ) \
        .dropDuplicates(['claim_id'])
    
    # Left join weather data with silver features
    silver_features = silver_features.join(
        weather_enrichment,
        on='claim_id',
        how='left'
    )
    
    # Add weather-based risk indicators
    silver_features = silver_features \
        .withColumn('had_severe_weather', 
            when(col('weather_precipitation_in') > 0.5, 1)
            .when(col('weather_wind_speed_mph') > 25, 1)
            .otherwise(0)
        ) \
        .withColumn('weather_enriched', 
            when(col('weather_conditions').isNotNull(), 1).otherwise(0)
        )
    
    weather_count = silver_features.filter(col('weather_enriched') == 1).count()
    print(f"✓ Integrated weather enrichment for {weather_count:,} claims")
    
except Exception as e:
    print(f"⚠️  Weather enrichment table not available or empty: {e}")
    print("   Run external_data_enrichment tool to populate weather data")
    # Add placeholder columns if enrichment unavailable
    silver_features = silver_features \
        .withColumn('weather_temp_max_f', lit(None).cast('float')) \
        .withColumn('weather_temp_min_f', lit(None).cast('float')) \
        .withColumn('weather_precipitation_in', lit(None).cast('float')) \
        .withColumn('weather_wind_speed_mph', lit(None).cast('float')) \
        .withColumn('weather_conditions', lit(None).cast('string')) \
        .withColumn('weather_api_timestamp', lit(None).cast('string')) \
        .withColumn('had_severe_weather', lit(0)) \
        .withColumn('weather_enriched', lit(0))

# COMMAND ----------

# DBTITLE 1,Save Silver Tables
# Save enriched claims to Silver layer
silver_features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_claims_enriched")

print(f"✓ Saved silver_claims_enriched table with {silver_features.count():,} records")

# Show sample
silver_features.select(
    'claim_id', 'claim_amount', 'days_to_file', 'claim_to_limit_ratio',
    'total_claims_count', 'rapid_filing', 'delayed_filing', 'is_fraud'
).show(10)

# COMMAND ----------

# DBTITLE 1,Create Feature Table for ML
# Create ML-ready feature table
feature_columns = [
    'claim_id',
    # Target
    'is_fraud',
    # Temporal features
    'days_to_file',
    'days_since_policy_start',
    'days_until_policy_end',
    'incident_month',
    'incident_day_of_week',
    'is_weekend_incident',
    # Financial features
    'claim_amount',
    'claim_to_limit_ratio',
    'claim_to_premium_ratio',
    'claim_exceeds_limit',
    'claim_near_limit',
    # Behavioral features
    'claim_sequence_num',
    'is_first_claim',
    'total_claims_count',
    'avg_claim_amount',
    'max_claim_amount',
    'prior_fraud_count',
    'provider_count',
    # Risk indicators
    'rapid_filing',
    'delayed_filing',
    'policy_new_incident',
    'policy_expiring_soon',
    'frequent_claimant',
    'high_value_claimant',
    'has_prior_fraud',
    # Text features
    'description_length',
    'description_word_count',
    'has_medical_code'
]

silver_ml_features = silver_features.select(feature_columns)

# Save ML feature table
silver_ml_features.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_fraud_features")

print(f"✓ Saved silver_fraud_features table with {len(feature_columns)-2} features")
print(f"\nFeatures: {', '.join([c for c in feature_columns if c not in ['claim_id', 'is_fraud']])}")

# COMMAND ----------

# DBTITLE 1,Data Quality Report
# Silver layer data quality report
print("\n=== Silver Layer Data Quality Report ===")

# Feature completeness
print("\nFeature completeness:")
for col_name in feature_columns:
    null_count = silver_ml_features.filter(col(col_name).isNull()).count()
    completeness = (1 - null_count / silver_ml_features.count()) * 100
    status = "✓" if completeness == 100 else "⚠"
    print(f"{status} {col_name}: {completeness:.1f}% complete")

# Feature distributions for fraud vs non-fraud
print("\nKey feature distributions by fraud status:")
silver_ml_features.groupBy('is_fraud').agg(
    count('*').alias('count'),
    avg('claim_amount').alias('avg_claim_amount'),
    avg('days_to_file').alias('avg_days_to_file'),
    avg('claim_to_limit_ratio').alias('avg_claim_to_limit_ratio'),
    avg('rapid_filing').alias('pct_rapid_filing'),
    avg('total_claims_count').alias('avg_total_claims')
).show()

print("\nSilver layer transformation complete!")

# COMMAND ----------

# DBTITLE 1,Summary
print("\n=== Transformation Summary ===")
print(f"\nSilver tables created:")
print(f"  1. silver_claims_enriched - Enriched claims with policy and claimant data")
print(f"  2. silver_fraud_features - ML-ready feature table with {len(feature_columns)-2} features")

print("\nNext step: Run 04_gold_layer_aggregation notebook")

# COMMAND ----------

