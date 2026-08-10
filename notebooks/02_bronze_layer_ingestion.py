# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Bronze Layer - Raw Data Ingestion
# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer - Raw Data Ingestion
# MAGIC 
# MAGIC This notebook validates and persists raw insurance claims data into the Bronze layer.
# MAGIC 
# MAGIC ## Bronze Layer Characteristics:
# MAGIC * Raw, unprocessed data
# MAGIC * Schema validation
# MAGIC * Metadata tracking (ingestion timestamp, source)
# MAGIC * Change Data Capture (CDC) ready
# MAGIC * Full audit trail

# COMMAND ----------

# DBTITLE 1,Configuration
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime

CATALOG = "main"
SCHEMA = "insurance_claims"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Working with: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Validate Bronze Tables
# Validate that bronze tables exist from data generation
tables_to_validate = ['bronze_claims', 'bronze_policies', 'bronze_claimants']

for table in tables_to_validate:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{table}").count()
    print(f"✓ {table}: {count:,} records")

print("\nBronze layer validation complete")

# COMMAND ----------

# DBTITLE 1,Add Metadata and Audit Columns
# Enrich bronze tables with metadata
from pyspark.sql.functions import current_timestamp, lit, input_file_name

for table in tables_to_validate:
    df = spark.table(f"{CATALOG}.{SCHEMA}.{table}")
    
    # Add metadata if not already present
    if '_ingestion_timestamp' not in df.columns:
        df = df.withColumn('_ingestion_timestamp', current_timestamp())
    if '_source_system' not in df.columns:
        df = df.withColumn('_source_system', lit('synthetic_generator'))
    if '_record_hash' not in df.columns:
        # Create hash for CDC tracking
        df = df.withColumn('_record_hash', md5(concat_ws('||', *df.columns)))
    
    # Overwrite with enriched data
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.{table}")
    print(f"✓ Enriched {table} with metadata columns")

print("\nMetadata enrichment complete")

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Bronze layer data quality checks
print("=== Data Quality Checks ===")

# Check 1: No duplicate claim IDs
claims_df = spark.table(f"{CATALOG}.{SCHEMA}.bronze_claims")
duplicate_claims = claims_df.groupBy('claim_id').count().filter(col('count') > 1)
dup_count = duplicate_claims.count()
print(f"\nDuplicate claim IDs: {dup_count} {'✓' if dup_count == 0 else '✗'}")

# Check 2: All claims have valid policy references
claims_without_policy = claims_df.join(
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_policies"),
    'policy_id',
    'left_anti'
).count()
print(f"Claims without valid policy: {claims_without_policy} {'✓' if claims_without_policy == 0 else '✗'}")

# Check 3: Filing dates after incident dates
invalid_dates = claims_df.filter(col('filing_date') < col('incident_date')).count()
print(f"Invalid date sequences: {invalid_dates} {'✓' if invalid_dates == 0 else '✗'}")

# Check 4: Claim amounts are positive
invalid_amounts = claims_df.filter(col('claim_amount') <= 0).count()
print(f"Invalid claim amounts: {invalid_amounts} {'✓' if invalid_amounts == 0 else '✗'}")

print("\nData quality checks complete")

# COMMAND ----------

# DBTITLE 1,Summary
print("\n=== Bronze Layer Ingestion Complete ===")
print(f"\nTables in {CATALOG}.{SCHEMA}:")
spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").show(truncate=False)

print("\nNext step: Run 03_silver_layer_transformation notebook")

# COMMAND ----------

