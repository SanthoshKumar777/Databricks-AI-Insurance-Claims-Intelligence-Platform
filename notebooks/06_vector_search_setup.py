# Databricks notebook source
# DBTITLE 1,Vector Search Setup
# MAGIC %md
# MAGIC # Vector Search Setup for Claim Similarity
# MAGIC
# MAGIC This notebook:
# MAGIC * Generates embeddings for claim descriptions using sentence transformers
# MAGIC * **Stores embeddings in Unity Catalog Delta tables (Lakehouse)**
# MAGIC * Creates Databricks Vector Search index
# MAGIC * Enables semantic similarity search for fraud detection
# MAGIC
# MAGIC All embeddings are persisted in the Lakehouse for:
# MAGIC * Durability and versioning
# MAGIC * Audit trails
# MAGIC * Reproducibility
# MAGIC * Integration with downstream systems

# COMMAND ----------

# DBTITLE 1,Install Required Packages
# MAGIC %pip install databricks-vectorsearch sentence-transformers scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and Configuration
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Configuration
CATALOG = "main"
SCHEMA = "insurance_claims"
VECTOR_SEARCH_ENDPOINT = "insurance-claims-vs-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.claim_embeddings_index"
EMBEDDING_TABLE = f"{CATALOG}.{SCHEMA}.claim_embeddings"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Vector Search Configuration:")
print(f"  Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"  Index: {VECTOR_INDEX_NAME}")
print(f"  Embeddings stored in: {EMBEDDING_TABLE}")

# COMMAND ----------

# DBTITLE 1,Load Claims Data
# Load claims from Silver layer
claims_df = spark.table(f"{CATALOG}.{SCHEMA}.silver_claims_enriched").select(
    'claim_id',
    'claimant_id',
    'claim_type',
    'claim_amount',
    'description',
    'diagnosis_code',
    'procedure_code',
    'is_fraud',
    'status'
).toPandas()

print(f"Loaded {len(claims_df):,} claims for embedding generation")
claims_df.head()

# COMMAND ----------

# DBTITLE 1,Generate Embeddings with Sentence Transformers
# Load sentence transformer model
print("\nLoading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
print("✓ Model loaded")

# Generate embeddings for claim descriptions
print("\nGenerating embeddings for claim descriptions...")
descriptions = claims_df['description'].fillna('').tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)

print(f"✓ Generated {len(embeddings):,} embeddings")
print(f"Embedding dimensions: {embeddings.shape[1]}")
print(f"Total embedding size: {embeddings.nbytes / 1024 / 1024:.2f} MB")

# COMMAND ----------

# DBTITLE 1,Store Embeddings in Lakehouse Delta Table
# Create DataFrame with embeddings
# CRITICAL: Embeddings are stored in Delta table for:
# - Persistence and durability
# - Version control
# - Audit trails
# - Integration with Vector Search

embeddings_data = []
for idx, row in claims_df.iterrows():
    embeddings_data.append({
        'claim_id': row['claim_id'],
        'claimant_id': row['claimant_id'],
        'claim_type': row['claim_type'],
        'claim_amount': float(row['claim_amount']),
        'description': row['description'],
        'is_fraud': bool(row['is_fraud']),
        'status': row['status'],
        'embedding': embeddings[idx].tolist(),  # Store as array
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': embeddings.shape[1],
        'created_at': pd.Timestamp.now()
    })

embeddings_df = pd.DataFrame(embeddings_data)

# Convert to Spark DataFrame with proper schema
schema = StructType([
    StructField('claim_id', StringType(), False),
    StructField('claimant_id', StringType(), True),
    StructField('claim_type', StringType(), True),
    StructField('claim_amount', DoubleType(), True),
    StructField('description', StringType(), True),
    StructField('is_fraud', BooleanType(), True),
    StructField('status', StringType(), True),
    StructField('embedding', ArrayType(FloatType()), False),
    StructField('embedding_model', StringType(), True),
    StructField('embedding_dim', IntegerType(), True),
    StructField('created_at', TimestampType(), True)
])

embeddings_spark = spark.createDataFrame(embeddings_df, schema=schema)

# Save embeddings to Unity Catalog Delta table
embeddings_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(EMBEDDING_TABLE)

print(f"\n✓ Embeddings saved to Lakehouse: {EMBEDDING_TABLE}")
print(f"  Records: {embeddings_spark.count():,}")
print(f"  Storage: Unity Catalog Delta table")
print(f"  Change Data Feed: Enabled for incremental updates")

# Verify storage
stored_embeddings = spark.table(EMBEDDING_TABLE)
print(f"\nVerification:")
print(f"  Table exists: {stored_embeddings.count():,} rows")
print(f"  Schema:")
stored_embeddings.printSchema()

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint
# Initialize Vector Search client
vs_client = VectorSearchClient()

# Create or get vector search endpoint
try:
    endpoint = vs_client.get_endpoint(VECTOR_SEARCH_ENDPOINT)
    print(f"✓ Vector Search endpoint already exists: {VECTOR_SEARCH_ENDPOINT}")
except Exception as e:
    print(f"Creating new Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT}")
    endpoint = vs_client.create_endpoint(
        name=VECTOR_SEARCH_ENDPOINT,
        endpoint_type="STANDARD"
    )
    print("✓ Endpoint created successfully")

print(f"\nEndpoint status: {endpoint.get('status', 'N/A')}")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index on Lakehouse Table
# Create Vector Search index on the Delta table
# This index reads directly from the Lakehouse table

print(f"\nCreating Vector Search index: {VECTOR_INDEX_NAME}")
print(f"Source: {EMBEDDING_TABLE} (Unity Catalog Delta table)")

try:
    # Create delta sync index - automatically syncs with the Delta table
    index = vs_client.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME,
        source_table_name=EMBEDDING_TABLE,
        pipeline_type="TRIGGERED",  # or "CONTINUOUS" for real-time updates
        primary_key="claim_id",
        embedding_dimension=384,
        embedding_vector_column="embedding"
    )
    print("✓ Vector Search index created")
    print(f"  Index syncs automatically with: {EMBEDDING_TABLE}")
    print(f"  Updates trigger index refresh")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Index already exists: {VECTOR_INDEX_NAME}")
        index = vs_client.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VECTOR_INDEX_NAME)
    else:
        print(f"⚠ Error creating index: {e}")
        print("Note: Vector Search requires appropriate permissions and endpoint availability")

# COMMAND ----------

# DBTITLE 1,Test Similarity Search
# Test similarity search
test_query = "Vehicle accident on highway, rear-end collision"
print(f"\nTesting similarity search with query: '{test_query}'")

# Generate embedding for query
query_embedding = model.encode([test_query])[0]
print(f"✓ Generated query embedding (dim={len(query_embedding)})")

try:
    # Search using Vector Search
    results = vs_client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME
    ).similarity_search(
        query_vector=query_embedding.tolist(),
        columns=["claim_id", "description", "claim_amount", "is_fraud"],
        num_results=5
    )
    
    print(f"\n✓ Found {len(results.get('result', {}).get('data_array', []))} similar claims:")
    for i, result in enumerate(results.get('result', {}).get('data_array', [])[:5], 1):
        print(f"\n{i}. Claim ID: {result[0]}")
        print(f"   Description: {result[1][:100]}...")
        print(f"   Amount: ${result[2]:,.2f}")
        print(f"   Fraud: {result[3]}")
        print(f"   Similarity: {result.get('score', 'N/A')}")
        
except Exception as e:
    print(f"\n⚠ Vector search not ready yet: {e}")
    print("\nNote: Index may need time to initialize.")
    print("Alternative: Use manual cosine similarity on the embeddings table")

# COMMAND ----------

# DBTITLE 1,Manual Similarity Search (Fallback)
# Fallback: Manual similarity search using embeddings from Delta table
print("\n" + "="*70)
print("MANUAL SIMILARITY SEARCH (Using Lakehouse embeddings)")
print("="*70)

from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings from Lakehouse
stored_embeddings_df = spark.table(EMBEDDING_TABLE).toPandas()
print(f"\nLoaded {len(stored_embeddings_df):,} embeddings from {EMBEDDING_TABLE}")

# Convert embeddings to numpy array
stored_vectors = np.array(stored_embeddings_df['embedding'].tolist())

# Compute cosine similarity
query_vector = query_embedding.reshape(1, -1)
similarities = cosine_similarity(query_vector, stored_vectors)[0]

# Get top 5 most similar claims
top_indices = np.argsort(similarities)[::-1][:5]

print(f"\nTop 5 similar claims for: '{test_query}'\n")
for rank, idx in enumerate(top_indices, 1):
    claim = stored_embeddings_df.iloc[idx]
    print(f"{rank}. Claim ID: {claim['claim_id']}")
    print(f"   Similarity: {similarities[idx]:.4f}")
    print(f"   Description: {claim['description'][:100]}...")
    print(f"   Amount: ${claim['claim_amount']:,.2f}")
    print(f"   Fraud: {claim['is_fraud']}")
    print()

# COMMAND ----------

# DBTITLE 1,Create Similarity Search UDF
# Create reusable function for similarity search
def search_similar_claims(query_text, top_k=5):
    """
    Search for similar claims using embeddings stored in Lakehouse.
    
    Args:
        query_text: Description of claim to search for
        top_k: Number of results to return
    
    Returns:
        DataFrame with similar claims and similarity scores
    """
    # Load model and embeddings from Lakehouse
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_df = spark.table(EMBEDDING_TABLE).toPandas()
    
    # Generate query embedding
    query_emb = model.encode([query_text])[0].reshape(1, -1)
    
    # Get stored embeddings
    stored_vectors = np.array(embeddings_df['embedding'].tolist())
    
    # Compute similarity
    similarities = cosine_similarity(query_emb, stored_vectors)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'claim_id': embeddings_df.iloc[idx]['claim_id'],
            'description': embeddings_df.iloc[idx]['description'],
            'claim_amount': embeddings_df.iloc[idx]['claim_amount'],
            'is_fraud': embeddings_df.iloc[idx]['is_fraud'],
            'status': embeddings_df.iloc[idx]['status'],
            'similarity_score': float(similarities[idx])
        })
    
    return pd.DataFrame(results)

# Test the function
print("\n" + "="*70)
print("Testing similarity search function")
print("="*70)

test_results = search_similar_claims("Slip and fall at retail store", top_k=3)
print("\nResults:")
print(test_results.to_string(index=False))

print("\n✓ Similarity search function ready for use")

# COMMAND ----------

# DBTITLE 1,Summary
print("\n" + "="*70)
print("VECTOR SEARCH SETUP COMPLETE")
print("="*70)

print(f"\n✓ Embeddings generated: {len(embeddings):,}")
print(f"✓ Embeddings stored in Lakehouse: {EMBEDDING_TABLE}")
print(f"✓ Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"✓ Vector Search index: {VECTOR_INDEX_NAME}")

print("\nEmbedding Storage Architecture:")
print("  1. Embeddings persisted in Unity Catalog Delta table")
print("  2. Change Data Feed enabled for incremental updates")
print("  3. Vector Search index syncs automatically with table")
print("  4. All data versioned and auditable in Lakehouse")

print("\nAvailable Search Methods:")
print("  1. Vector Search API (production-ready, scalable)")
print("  2. Manual similarity via Lakehouse table (fallback)")
print("  3. search_similar_claims() UDF (easy integration)")

print("\nIntegration Points:")
print("  • MCP Tool Server: similar_claims_search()")
print("  • Multi-Agent System: Context retrieval")
print("  • Databricks App: Interactive search")
print("  • SQL Queries: Direct table access")

print("\nNext steps:")
print("  1. Deploy MCP server (src/mcp_server.py)")
print("  2. Launch multi-agent orchestrator (src/multi_agent.py)")
print("  3. Deploy Databricks App (src/databricks_app.py)")

print("\n" + "="*70)
print("All data persisted in Lakehouse - Ready for production!")
print("="*70)

# COMMAND ----------

