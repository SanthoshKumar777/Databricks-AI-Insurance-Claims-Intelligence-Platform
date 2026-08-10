# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup and Configuration
# Databricks notebook source
# MAGIC %md
# MAGIC # Insurance Claims Synthetic Data Generation
# MAGIC 
# MAGIC This notebook generates realistic insurance claims data with embedded fraud patterns.
# MAGIC 
# MAGIC ## Features:
# MAGIC * Realistic claim distributions
# MAGIC * Multiple fraud patterns (billing inflation, staged accidents, duplicate claims)
# MAGIC * Policy and claimant information
# MAGIC * Medical codes (ICD-10, CPT)
# MAGIC * Geospatial data

# COMMAND ----------

# DBTITLE 1,Import Libraries
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import uuid
import builtins

# Preserve Python's built-ins (PySpark's import * overwrites them)
python_round = builtins.round
python_min = builtins.min
python_max = builtins.max

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# COMMAND ----------

# DBTITLE 1,Configuration Parameters
# Configuration
CATALOG = "main"
SCHEMA = "insurance_claims"
NUM_CLAIMS = 10000
FRAUD_RATE = 0.08  # 8% fraud rate

# Create catalog and schema if they don't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Generating {NUM_CLAIMS} claims with {FRAUD_RATE*100}% fraud rate")
print(f"Target location: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Helper Functions and Data Definitions
# Medical codes (ICD-10 diagnosis codes)
COMMON_DIAGNOSES = {
    'S06.9': 'Head injury, unspecified',
    'S83.5': 'Sprain of knee',
    'M54.5': 'Low back pain',
    'S52.5': 'Fracture of lower end of radius',
    'T14.9': 'Injury, unspecified',
    'S43.4': 'Sprain of shoulder joint',
    'S93.4': 'Sprain of ankle',
    'M79.3': 'Panniculitis, unspecified',
    'S72.0': 'Fracture of femur',
    'S82.5': 'Fracture of tibia'
}

# CPT procedure codes
COMMON_PROCEDURES = {
    '99285': 'Emergency department visit, high severity',
    '73610': 'Radiologic examination, ankle',
    '29881': 'Arthroscopy, knee, surgical',
    '20610': 'Arthrocentesis, major joint',
    '97110': 'Therapeutic exercises',
    '72148': 'MRI lumbar spine',
    '97140': 'Manual therapy techniques',
    '29125': 'Application of short arm splint',
    '27447': 'Total knee arthroplasty',
    '73562': 'Radiologic examination, knee'
}

# US States
STATES = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']

# Insurance providers
INSURANCE_PROVIDERS = ['BlueCross', 'Aetna', 'UnitedHealth', 'Cigna', 'Humana', 'Kaiser']

# Claim types
CLAIM_TYPES = ['Auto', 'Property', 'Health', 'Workers Comp', 'Liability']

def random_date(start_date, end_date):
    """Generate random date between start and end"""
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)

def generate_claim_amount(claim_type, is_fraud=False):
    """Generate realistic claim amounts"""
    base_amounts = {
        'Auto': (2000, 15000),
        'Property': (5000, 50000),
        'Health': (1000, 25000),
        'Workers Comp': (3000, 40000),
        'Liability': (10000, 100000)
    }
    
    min_amt, max_amt = base_amounts[claim_type]
    amount = random.uniform(min_amt, max_amt)
    
    # Inflate fraudulent claims
    if is_fraud:
        inflation_factor = random.uniform(1.3, 2.5)
        amount *= inflation_factor
    
    return python_round(amount, 2)

# COMMAND ----------

# DBTITLE 1,Generate Claimants
# Generate claimant data
num_claimants = NUM_CLAIMS // 2  # Some claimants have multiple claims

claimants_data = []
for i in range(num_claimants):
    claimant_id = f"CLT-{str(i+1).zfill(6)}"
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa', 'James', 'Mary']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    
    claimants_data.append({
        'claimant_id': claimant_id,
        'first_name': random.choice(first_names),
        'last_name': random.choice(last_names),
        'date_of_birth': random_date(datetime(1950, 1, 1), datetime(2000, 12, 31)).strftime('%Y-%m-%d'),
        'email': f"claimant{i+1}@email.com",
        'phone': f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}",
        'address_state': random.choice(STATES),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

claimants_df = spark.createDataFrame(claimants_data)
claimants_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_claimants")

print(f"Generated {num_claimants} claimants")
claimants_df.show(5)

# COMMAND ----------

# DBTITLE 1,Generate Policies
# Generate policy data
policies_data = []

for i in range(NUM_CLAIMS):
    policy_id = f"POL-{str(i+1).zfill(6)}"
    claim_type = random.choice(CLAIM_TYPES)
    
    # Set policy limits based on type
    limit_ranges = {
        'Auto': (25000, 100000),
        'Property': (100000, 500000),
        'Health': (50000, 1000000),
        'Workers Comp': (100000, 500000),
        'Liability': (250000, 2000000)
    }
    
    min_limit, max_limit = limit_ranges[claim_type]
    policy_limit = random.choice([min_limit, max_limit//2, max_limit])
    
    policy_start = random_date(datetime(2020, 1, 1), datetime(2023, 1, 1))
    policy_end = policy_start + timedelta(days=365)
    
    deductible_amt = random.choice([500, 1000, 2500, 5000])
    premium = python_round(policy_limit * 0.015 * random.uniform(0.8, 1.2), 2)
    
    policies_data.append({
        'policy_id': policy_id,
        'claimant_id': claimants_data[i % num_claimants]['claimant_id'],
        'policy_type': claim_type,
        'provider': random.choice(INSURANCE_PROVIDERS),
        'policy_limit': float(policy_limit),
        'deductible': float(deductible_amt),
        'premium_annual': float(premium),
        'start_date': policy_start.strftime('%Y-%m-%d'),
        'end_date': policy_end.strftime('%Y-%m-%d'),
        'status': 'Active',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

policies_df = spark.createDataFrame(policies_data)
policies_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_policies")

print(f"Generated {len(policies_data)} policies")
policies_df.show(5)

# COMMAND ----------

# DBTITLE 1,Generate Claims with Fraud Patterns
# Generate claims data with fraud patterns
claims_data = []

for i in range(NUM_CLAIMS):
    claim_id = f"CLM-{datetime.now().year}-{str(i+1).zfill(5)}"
    policy = policies_data[i]
    
    # Determine if this claim is fraudulent
    is_fraud = random.random() < FRAUD_RATE
    
    # Incident date within policy period
    policy_start = datetime.strptime(policy['start_date'], '%Y-%m-%d')
    policy_end = datetime.strptime(policy['end_date'], '%Y-%m-%d')
    incident_date = random_date(policy_start, python_min(policy_end, datetime.now()))
    
    # Filing date - fraudulent claims may be filed suspiciously quickly or slowly
    if is_fraud and random.random() < 0.3:
        # Fraud pattern: Filed very quickly (same day or next day)
        days_to_file = random.randint(0, 1)
    elif is_fraud and random.random() < 0.3:
        # Fraud pattern: Filed unusually late
        days_to_file = random.randint(60, 180)
    else:
        # Normal filing pattern
        days_to_file = random.randint(1, 30)
    
    filing_date = incident_date + timedelta(days=days_to_file)
    
    # Generate claim amount
    claim_amount = generate_claim_amount(policy['policy_type'], is_fraud)
    
    # Fraud pattern: Claim exceeds or nearly equals policy limit
    if is_fraud and random.random() < 0.4:
        claim_amount = policy['policy_limit'] * random.uniform(0.95, 1.0)
    
    # Medical codes for health-related claims
    diagnosis_code = random.choice(list(COMMON_DIAGNOSES.keys()))
    procedure_code = random.choice(list(COMMON_PROCEDURES.keys()))
    
    # Claim description
    descriptions = [
        f"Vehicle accident on highway, {random.choice(['rear-end', 'side-impact', 'head-on'])} collision",
        f"Slip and fall incident at {random.choice(['workplace', 'retail store', 'parking lot'])}",
        f"Property damage due to {random.choice(['fire', 'water damage', 'storm', 'theft'])}",
        f"Medical treatment for {COMMON_DIAGNOSES[diagnosis_code]}",
        f"Work-related injury requiring {COMMON_PROCEDURES[procedure_code]}"
    ]
    
    # Fraud pattern: Duplicate or very similar claims
    if is_fraud and random.random() < 0.2 and i > 100:
        # Copy description from a previous claim
        description = claims_data[random.randint(0, python_min(i-1, 100))]['description']
    else:
        description = random.choice(descriptions)
    
    # Status based on fraud and claim characteristics
    if is_fraud and random.random() < 0.3:
        status = random.choice(['Under Investigation', 'Denied'])
    else:
        status = random.choice(['Submitted', 'Under Review', 'Approved', 'Paid', 'Closed'])
    
    claims_data.append({
        'claim_id': claim_id,
        'policy_id': policy['policy_id'],
        'claimant_id': policy['claimant_id'],
        'claim_type': policy['policy_type'],
        'incident_date': incident_date.strftime('%Y-%m-%d'),
        'filing_date': filing_date.strftime('%Y-%m-%d'),
        'claim_amount': claim_amount,
        'description': description,
        'diagnosis_code': diagnosis_code if policy['policy_type'] in ['Health', 'Workers Comp'] else None,
        'procedure_code': procedure_code if policy['policy_type'] in ['Health', 'Workers Comp'] else None,
        'incident_state': next((c['address_state'] for c in claimants_data if c['claimant_id'] == policy['claimant_id']), 'CA'),  # Look up actual claimant state
        'status': status,
        'is_fraud': is_fraud,
        'days_to_file': days_to_file,
        'claim_to_limit_ratio': python_round(claim_amount / policy['policy_limit'], 4),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

claims_df = spark.createDataFrame(claims_data)
claims_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_claims")

print(f"Generated {NUM_CLAIMS} claims")
print(f"Fraudulent claims: {claims_df.filter(col('is_fraud') == True).count()}")
print(f"Legitimate claims: {claims_df.filter(col('is_fraud') == False).count()}")
claims_df.show(5)

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Data quality checks
print("\n=== Data Quality Summary ===")

# Check for nulls
print("\nNull counts in claims:")
claims_df.select([count(when(col(c).isNull(), c)).alias(c) for c in claims_df.columns]).show()

# Fraud distribution
print("\nFraud distribution:")
claims_df.groupBy('is_fraud').agg(
    count('*').alias('count'),
    avg('claim_amount').alias('avg_amount'),
    avg('days_to_file').alias('avg_days_to_file'),
    avg('claim_to_limit_ratio').alias('avg_claim_to_limit_ratio')
).show()

# Status distribution
print("\nClaim status distribution:")
claims_df.groupBy('status').count().orderBy(desc('count')).show()

# Claim type distribution
print("\nClaim type distribution:")
claims_df.groupBy('claim_type').agg(
    count('*').alias('count'),
    avg('claim_amount').alias('avg_amount'),
    sum('claim_amount').alias('total_amount')
).show()

# COMMAND ----------

# DBTITLE 1,Summary Statistics
# Summary statistics
print("\n=== Generation Complete ===")
print(f"\nTables created in {CATALOG}.{SCHEMA}:")
print(f"  - bronze_claimants ({num_claimants} records)")
print(f"  - bronze_policies ({len(policies_data)} records)")
print(f"  - bronze_claims ({NUM_CLAIMS} records)")

print("\nFraud patterns embedded:")
print("  ✓ Billing inflation (1.3x - 2.5x normal amounts)")
print("  ✓ Rapid filing (same-day or next-day claims)")
print("  ✓ Late filing (60-180 days after incident)")
print("  ✓ Policy limit exploitation (95-100% of limit)")
print("  ✓ Duplicate claim descriptions")

print("\nNext steps:")
print("  1. Run 02_bronze_layer_ingestion notebook")
print("  2. Run 03_silver_layer_transformation notebook")
print("  3. Run 04_gold_layer_aggregation notebook")
print("  4. Train fraud detection model in 05_ml_fraud_detection notebook")

# COMMAND ----------

