# Databricks AI Insurance Claims Intelligence Platform - Deployment Guide

## 📁 Project Structure

```
Databricks-AI-Insurance-Claims-Intelligence-Platform/
├── app/                          # Streamlit application
│   ├── __init__.py              # Python package marker
│   └── app.py                   # Main Streamlit app (entry point)
│
├── src/                          # Source code modules
│   ├── __init__.py              # Python package marker
│   ├── mcp_server.py            # MCP Tool Server (7 investigation tools)
│   ├── multi_agent.py           # Multi-agent orchestration system
│   ├── evaluation_framework.py  # Model evaluation framework
│   └── databricks_app.py        # Legacy app file (deprecated)
│
├── notebooks/                    # Data pipeline notebooks
│   ├── 01_generate_synthetic_data.py
│   ├── 02_bronze_layer_ingestion.py
│   ├── 03_silver_layer_transformation.py
│   ├── 04_gold_layer_aggregation.py
│   ├── 05_ml_fraud_detection.py
│   └── 06_vector_search_setup.py
│
├── app.yaml                      # Databricks App configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🔧 Recent Fixes

### ✅ Import Errors Fixed

**Problem:** `ModuleNotFoundError: No module named 'src'` and `cannot import name 'sql' from 'databricks'`

**Solutions Applied:**
1. Created proper `__init__.py` files in `app/` and `src/` folders
2. Updated import path in `app/app.py`:
   ```python
   sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
   from mcp_server import MCPToolServer
   ```
3. Removed unused `from databricks import sql` import from `mcp_server.py`
4. Updated `app.yaml` to point to correct entry point: `app/app.py`

### ✅ app.yaml Format Error Fixed (CRITICAL)

**Problem:** "Error reading app.yaml file, please ensure it is in the correct format"

**Root Cause:** app.yaml had too many unsupported fields (74 lines with app_name, compute, environment.dependencies, resources, ports, health_check, logging, scaling, tags). Databricks Apps requires a MINIMAL format.

**Solution Applied:**
1. **Removed all unsupported fields** - Only kept `command` and `env`
2. **Fixed env format** - Changed to list format: `[{name: X, value: Y}]`
3. **Moved dependencies** - All Python packages now in `requirements.txt`
4. **Minimal YAML** - Reduced from 74 lines to 11 lines

Final app.yaml (MINIMAL format):
```yaml
command:
  - streamlit
  - run
  - app/app.py
  - --server.port=8080
  - --server.address=0.0.0.0

env:
  - name: CATALOG
    value: main
  - name: SCHEMA
    value: insurance_claims
```

### ✅ App Startup Crash Fixed (CRITICAL)

**Problem:** App exited unexpectedly during deployment with "app exited unexpectedly" error

**Root Cause:** `MCPToolServer` was trying to initialize `WorkspaceClient()` and create `SparkSession` immediately on import, which failed during the app build phase.

**Solutions Applied:**
1. **Lazy initialization**: Changed `WorkspaceClient` to load only when needed via `@property`
2. **Graceful fallback**: `_execute_sql()` now falls back to mock data if Spark is unavailable
3. **Error handling**: Wrapped MCP server initialization in try-except in `app.py`
4. **Demo mode**: App runs with mock data if database connection fails
5. **User feedback**: Added warning banner when running in demo mode

### ✅ Project Structure Reorganized

- Moved Streamlit app to `app/` folder
- Consolidated MCP server code to `src/` folder
- Removed duplicate `mcp/` folder
- Created Python package structure with `__init__.py` files

## 🚀 Deployment Instructions

### Option 1: Deploy as Databricks App (Recommended)

```bash
# From project root directory
databricks apps deploy
```

The deployment will:
- ✅ Read `app.yaml` configuration
- ✅ Install dependencies from `requirements.txt`
- ✅ Run: `streamlit run app/app.py --server.port=8080 --server.address=0.0.0.0`
- ✅ Set environment variables (CATALOG=main, SCHEMA=insurance_claims)

### Option 2: Run Locally for Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/app.py --server.port=8080 --server.address=0.0.0.0
```

### Option 3: Use Python Launcher

```bash
# The app.py has a built-in launcher
python app/app.py
```

## 📦 Dependencies

All dependencies are specified in `requirements.txt`:

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
databricks-sdk>=0.20.0
databricks-sql-connector>=3.0.0
sentence-transformers>=2.2.0
mlflow>=2.9.0
requests>=2.31.0
numpy>=1.24.0
```

## 🔍 Key Components

### 1. Streamlit App (`app/app.py`)
- Main web interface for claims investigation
- 5 pages: Dashboard, Claim Search, AI Investigation, Notes, Analytics
- Integrates with MCP Tool Server for investigation tools

### 2. MCP Tool Server (`src/mcp_server.py`)
Provides 7 specialized investigation tools:
1. **get_claim_details** - Comprehensive claim information
2. **fraud_risk_score** - ML-based fraud probability
3. **similar_claims_search** - Semantic similarity search
4. **policy_verification** - Policy coverage validation
5. **medical_code_lookup** - Diagnosis/procedure codes
6. **payment_history** - Claimant payment patterns
7. **external_data_enrichment** - Third-party data (weather, etc.)

### 3. Multi-Agent System (`src/multi_agent.py`)
4 specialized agents:
- **Fraud Detection Agent** - Pattern recognition
- **Legal Compliance Agent** - Regulatory compliance
- **Medical Review Agent** - Clinical documentation
- **Financial Analysis Agent** - Cost analysis

## 🛠️ Configuration

### Environment Variables (app.yaml)

```yaml
env:
  - name: CATALOG
    value: main
  - name: SCHEMA
    value: insurance_claims
```

### Compute Configuration

```yaml
compute:
  type: serverless  # Auto-scaling serverless compute
```

### Health Check

```yaml
health_check:
  path: /_stcore/health
  port: 8080
  initial_delay_seconds: 30
```

## ✅ Verification Steps

1. **Check imports work:**
   ```python
   import sys
   sys.path.insert(0, 'src')
   from mcp_server import MCPToolServer
   # Should succeed without errors
   ```

2. **Verify file structure:**
   ```bash
   ls -la app/ src/ *.yaml *.txt
   ```

3. **Test Streamlit app locally:**
   ```bash
   streamlit run app/app.py
   ```

## 🐛 Troubleshooting

### Issue: "Error reading app.yaml file, please ensure it is in the correct format"
**Root Cause:** app.yaml has unsupported fields or incorrect format
**Solution:** 
- ✅ FIXED: Created minimal app.yaml with ONLY `command` and `env` fields
- Databricks Apps requires a very simple format
- All dependencies must be in `requirements.txt` (NOT in app.yaml)
- Unsupported fields: app_name, compute, environment.dependencies, resources, ports, health_check, logging, scaling, tags
- Use the minimal format shown above (11 lines total)

### Issue: "App exited unexpectedly" during deployment
**Root Cause:** Server trying to initialize database connections during build phase
**Solution:** 
- ✅ FIXED: MCPToolServer now uses lazy initialization
- ✅ FIXED: App gracefully falls back to demo mode with mock data
- The app will start successfully even if database is unavailable
- Look for warning banner in the app UI indicating demo mode

### Issue: "ModuleNotFoundError: No module named 'mcp_server'"
**Solution:** Ensure `src/__init__.py` exists and `sys.path` is correctly set in `app/app.py`

### Issue: "Error reading app.yaml file"
**Solution:** Verify YAML syntax is correct. Use simple format without complex nested structures.

### Issue: "cannot import name 'sql' from 'databricks'"
**Solution:** Remove `from databricks import sql` - use PySpark SQL instead via SparkSession

### Issue: App running wrong entry point
**Solution:** Check `app.yaml` command points to `app/app.py`, not `app.py` or other files

### Issue: "property 'w' of 'MCPToolServer' object has no setter"
**Solution:** 
- ✅ FIXED: Removed direct assignment to `self.w` in `__init__`
- WorkspaceClient is now lazy-loaded via `@property`

## 📊 Data Pipeline Setup

Run notebooks in order to set up the data pipeline:

```bash
1. notebooks/01_generate_synthetic_data.py      # Generate synthetic claims data
2. notebooks/02_bronze_layer_ingestion.py       # Ingest raw data
3. notebooks/03_silver_layer_transformation.py  # Transform and enrich
4. notebooks/04_gold_layer_aggregation.py       # Create aggregations
5. notebooks/05_ml_fraud_detection.py           # Train fraud detection model
6. notebooks/06_vector_search_setup.py          # Set up semantic search
```

## 🎯 Success Criteria

✅ No import errors
✅ Streamlit app starts successfully
✅ MCP Tool Server loads and provides 7+ tools
✅ Database connections work (main.insurance_claims)
✅ All pages render correctly
✅ Investigation tools return results

## 📞 Support

For issues or questions:
1. Check this deployment guide
2. Review error logs in Databricks App UI
3. Verify all dependencies are installed
4. Ensure database schema exists (main.insurance_claims)

---

**Last Updated:** August 10, 2026
**Version:** 1.0.0
**Status:** ✅ All issues resolved, ready for deployment
