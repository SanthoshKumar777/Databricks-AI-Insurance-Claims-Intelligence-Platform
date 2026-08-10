#!/usr/bin/env python3
"""
Databricks AI Insurance Claims Intelligence Platform - Streamlit App

A comprehensive web application for insurance claims investigation featuring:
- Claim search and filtering
- Multi-agent AI investigation
- Semantic similarity search using embeddings
- Investigation notes management  
- ML model fraud scoring with explainability
- Analytics dashboards
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks import sql
from databricks.sdk import WorkspaceClient
import json
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mcp_server import MCPToolServer

# Page configuration
st.set_page_config(
    page_title="Insurance Claims Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #555;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .fraud-high {background-color: #ffebee; border-left: 5px solid #f44336;}
    .fraud-medium {background-color: #fff3e0; border-left: 5px solid #ff9800;}
    .fraud-low {background-color: #e8f5e9; border-left: 5px solid #4caf50;}
    .tool-result {background-color: #fafafa; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'mcp_server' not in st.session_state:
    st.session_state.mcp_server = MCPToolServer(catalog="main", schema="insurance_claims")

if 'investigation_history' not in st.session_state:
    st.session_state.investigation_history = []

if 'selected_claim' not in st.session_state:
    st.session_state.selected_claim = None

# Helper functions
@st.cache_data(ttl=300)
def load_claims_data(filters=None):
    """Load claims data with optional filters."""
    server = st.session_state.mcp_server
    query = f"SELECT * FROM {server.catalog}.{server.schema}.silver_claims_enriched"
    
    if filters:
        where_clauses = []
        if filters.get('status'):
            where_clauses.append(f"status = '{filters['status']}'")
        if filters.get('min_amount'):
            where_clauses.append(f"claim_amount >= {filters['min_amount']}")
        if filters.get('max_amount'):
            where_clauses.append(f"claim_amount <= {filters['max_amount']}")
        if filters.get('state'):
            where_clauses.append(f"address_state = '{filters['state']}'")
        if filters.get('fraud_only'):
            where_clauses.append("is_fraud = TRUE")
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ORDER BY filing_date DESC LIMIT 500"
    
    results = server._execute_sql(query)
    if results and 'error' not in results[0]:
        return pd.DataFrame(results)
    return pd.DataFrame()

@st.cache_data(ttl=600)
def load_analytics_data():
    """Load aggregated analytics data."""
    server = st.session_state.mcp_server
    query = f"""
    SELECT 
        COUNT(*) as total_claims,
        SUM(claim_amount) as total_amount,
        AVG(claim_amount) as avg_amount,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
        SUM(CASE WHEN is_fraud THEN claim_amount ELSE 0 END) as fraud_amount,
        SUM(CASE WHEN status = 'Approved' THEN 1 ELSE 0 END) as approved_count,
        SUM(CASE WHEN status = 'Denied' THEN 1 ELSE 0 END) as denied_count,
        SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) as pending_count
    FROM {server.catalog}.{server.schema}.silver_claims_enriched
    """
    results = server._execute_sql(query)
    return results[0] if results else {}

def run_agent_investigation(claim_id):
    """Run multi-agent investigation on a claim."""
    server = st.session_state.mcp_server
    
    results = {
        "claim_id": claim_id,
        "timestamp": datetime.now().isoformat(),
        "tools_called": []
    }
    
    with st.spinner("🔍 Running multi-agent investigation..."):
        # Tool 1: Get claim details
        with st.expander("📋 Claim Details", expanded=True):
            claim_details = server.get_claim_details(claim_id)
            results["tools_called"].append({"tool": "get_claim_details", "result": claim_details})
            st.json(claim_details)
        
        # Tool 2: Fraud risk score
        with st.expander("⚠️ Fraud Risk Assessment", expanded=True):
            fraud_score = server.fraud_risk_score(claim_id)
            results["tools_called"].append({"tool": "fraud_risk_score", "result": fraud_score})
            
            if fraud_score.get('status') == 'success':
                risk_score = fraud_score['fraud_risk_score']
                risk_level = fraud_score['risk_level']
                
                # Risk gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Level", risk_level)
                with col2:
                    st.metric("Recommendation", fraud_score.get('recommendation'))
                with col3:
                    st.metric("Model", fraud_score.get('model_version', 'N/A'))
                
                if fraud_score.get('risk_factors'):
                    st.subheader("Risk Factors:")
                    for factor in fraud_score['risk_factors']:
                        st.write(f"• {factor}")
            else:
                st.error(fraud_score.get('message', 'Fraud scoring failed'))
        
        # Tool 3: Similar claims
        if claim_details.get('status') == 'success':
            description = claim_details.get('claim', {}).get('description', '')
            if description:
                with st.expander("🔎 Similar Historical Claims", expanded=False):
                    similar_claims = server.similar_claims_search(description, top_k=5)
                    results["tools_called"].append({"tool": "similar_claims_search", "result": similar_claims})
                    
                    if similar_claims.get('status') == 'success':
                        st.info(f"Search method: {similar_claims.get('search_method', 'N/A')}")
                        
                        for idx, claim in enumerate(similar_claims.get('similar_claims', []), 1):
                            with st.container():
                                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                                with col1:
                                    st.write(f"**{idx}. {claim['claim_id']}**")
                                with col2:
                                    st.write(f"Amount: ${claim['claim_amount']:,.2f}")
                                with col3:
                                    st.write(f"Similarity: {claim['similarity_score']:.2%}")
                                with col4:
                                    if claim['was_fraud']:
                                        st.error("⚠️ Fraud")
                                    else:
                                        st.success("✓ Valid")
                                st.caption(claim['description'][:150] + "...")
                                st.divider()
                    else:
                        st.warning(similar_claims.get('message', 'Similar claims search failed'))
        
        # Tool 4: Policy verification
        if claim_details.get('status') == 'success':
            policy_id = claim_details.get('policy', {}).get('id')
            claim_amount = claim_details.get('claim', {}).get('amount')
            incident_date = claim_details.get('claim', {}).get('incident_date')
            
            if policy_id and claim_amount and incident_date:
                with st.expander("📄 Policy Verification", expanded=False):
                    policy_check = server.policy_verification(policy_id, claim_amount, incident_date)
                    results["tools_called"].append({"tool": "policy_verification", "result": policy_check})
                    
                    if policy_check.get('status') == 'success':
                        if policy_check.get('policy_valid'):
                            st.success("✓ Policy is valid")
                        else:
                            st.error("⚠️ Policy validation issues detected")
                        
                        for validation in policy_check.get('validations', []):
                            if validation['status'] == 'PASS':
                                st.success(f"✓ {validation['check']}: {validation['message']}")
                            elif validation['status'] == 'WARN':
                                st.warning(f"⚠️ {validation['check']}: {validation['message']}")
                            else:
                                st.error(f"❌ {validation['check']}: {validation['message']}")
        
        # Tool 5: External enrichment
        with st.expander("🌐 External Data Enrichment", expanded=False):
            enrichment = server.external_data_enrichment(claim_id, data_sources=['weather'])
            results["tools_called"].append({"tool": "external_data_enrichment", "result": enrichment})
            
            if enrichment.get('status') == 'success':
                weather_data = enrichment.get('enriched_data', {}).get('weather', {})
                if weather_data:
                    st.subheader("Weather Conditions at Incident:")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Max Temp", f"{weather_data.get('temperature_max_f', 'N/A')}°F")
                    with col2:
                        st.metric("Min Temp", f"{weather_data.get('temperature_min_f', 'N/A')}°F")
                    with col3:
                        st.metric("Precipitation", f"{weather_data.get('precipitation_inches', 0):.2f} in")
                    with col4:
                        st.metric("Wind Speed", f"{weather_data.get('wind_speed_mph', 'N/A')} mph")
                    
                    st.info(f"Conditions: {weather_data.get('conditions_summary', 'N/A')}")
                    st.caption(f"Source: {weather_data.get('source', 'N/A')}")
    
    # Save to history
    st.session_state.investigation_history.append(results)
    return results

# Main app
st.markdown('<p class="main-header">🔍 Insurance Claims Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown("<p class="sub-header">AI-Powered Claims Investigation & Fraud Detection</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard", "🔍 Claim Search", "🤖 AI Investigation", "📝 Investigation Notes", "📊 Analytics"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.subheader("Filters")
    
    status_filter = st.selectbox(
        "Claim Status",
        ["All", "Pending", "Approved", "Denied", "Under Investigation"],
        key="status_filter"
    )
    
    amount_range = st.slider(
        "Claim Amount Range ($)",
        min_value=0,
        max_value=200000,
        value=(0, 200000),
        step=5000,
        key="amount_filter"
    )
    
    fraud_only = st.checkbox("Show Fraud Cases Only", key="fraud_filter")
    
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Build filters dict
filters = {
    'status': None if status_filter == "All" else status_filter,
    'min_amount': amount_range[0],
    'max_amount': amount_range[1],
    'fraud_only': fraud_only
}

# Dashboard Page
if page == "🏠 Dashboard":
    st.subheader("Executive Dashboard")
    
    # Load analytics
    analytics = load_analytics_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Claims",
            f"{analytics.get('total_claims', 0):,}",
            delta=None
        )
    with col2:
        st.metric(
            "Total Amount",
            f"${analytics.get('total_amount', 0):,.0f}",
            delta=None
        )
    with col3:
        fraud_rate = (analytics.get('fraud_count', 0) / analytics.get('total_claims', 1)) * 100
        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.1f}%",
            delta=None,
            delta_color="inverse"
        )
    with col4:
        st.metric(
            "Pending Claims",
            f"{analytics.get('pending_count', 0):,}",
            delta=None
        )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_data = {
            'Status': ['Approved', 'Denied', 'Pending'],
            'Count': [
                analytics.get('approved_count', 0),
                analytics.get('denied_count', 0),
                analytics.get('pending_count', 0)
            ]
        }
        fig1 = px.pie(
            status_data,
            values='Count',
            names='Status',
            title='Claim Status Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Fraud vs Valid
        fraud_data = {
            'Category': ['Valid Claims', 'Fraudulent Claims'],
            'Count': [
                analytics.get('total_claims', 0) - analytics.get('fraud_count', 0),
                analytics.get('fraud_count', 0)
            ]
        }
        fig2 = px.bar(
            fraud_data,
            x='Category',
            y='Count',
            title='Fraud Detection Summary',
            color='Category',
            color_discrete_map={'Valid Claims': 'green', 'Fraudulent Claims': 'red'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Recent high-risk claims
    st.subheader("⚠️ Recent High-Risk Claims")
    high_risk_query = f"""
    SELECT claim_id, claim_amount, status, rapid_filing, claim_to_limit_ratio, has_prior_fraud
    FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.silver_fraud_features
    WHERE (rapid_filing = 1 OR claim_to_limit_ratio > 0.8 OR has_prior_fraud = 1)
    ORDER BY claim_amount DESC
    LIMIT 10
    """
    high_risk_results = st.session_state.mcp_server._execute_sql(high_risk_query)
    if high_risk_results and 'error' not in high_risk_results[0]:
        st.dataframe(
            pd.DataFrame(high_risk_results),
            use_container_width=True,
            hide_index=True
        )

# Claim Search Page
elif page == "🔍 Claim Search":
    st.subheader("Claim Search & Filter")
    
    # Search box
    search_query = st.text_input(
        "🔍 Search by Claim ID, Description, or Claimant",
        placeholder="Enter search term...",
        key="search_query"
    )
    
    # Load claims
    claims_df = load_claims_data(filters)
    
    if not claims_df.empty:
        # Apply search filter
        if search_query:
            mask = claims_df.apply(
                lambda row: search_query.lower() in str(row).lower(),
                axis=1
            )
            claims_df = claims_df[mask]
        
        st.success(f"Found {len(claims_df)} claims")
        
        # Display claims table
        st.dataframe(
            claims_df[[
                'claim_id', 'claim_amount', 'status', 'incident_date',
                'claim_type', 'is_fraud', 'full_name', 'address_state'
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "claim_amount": st.column_config.NumberColumn(
                    "Amount",
                    format="$%d"
                ),
                "is_fraud": st.column_config.CheckboxColumn(
                    "Fraud?"
                )
            }
        )
        
        # Claim selection
        selected_claim_id = st.selectbox(
            "Select a claim to investigate:",
            claims_df['claim_id'].tolist(),
            key="selected_claim_dropdown"
        )
        
        if st.button("🤖 Launch AI Investigation", type="primary"):
            st.session_state.selected_claim = selected_claim_id
            st.rerun()
    else:
        st.info("No claims found matching the filters.")

# AI Investigation Page
elif page == "🤖 AI Investigation":
    st.subheader("Multi-Agent AI Investigation")
    
    if st.session_state.selected_claim:
        st.info(f"Investigating Claim: **{st.session_state.selected_claim}**")
        
        if st.button("🔄 Run Investigation", type="primary"):
            results = run_agent_investigation(st.session_state.selected_claim)
            st.success("✅ Investigation complete!")
            
            # Action buttons
            st.divider()
            st.subheader("📝 Investigation Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚨 Escalate for Review"):
                    result = st.session_state.mcp_server.update_claim_status(
                        st.session_state.selected_claim,
                        "Under Investigation",
                        "Escalated by AI investigation",
                        "ai_agent"
                    )
                    if result.get('status') == 'success':
                        st.success("✅ Status updated!")
                        st.json(result)
            
            with col2:
                note_text = st.text_area("Investigation Note:", key="inv_note")
                if st.button("💾 Add Note"):
                    if note_text:
                        result = st.session_state.mcp_server.add_investigation_note(
                            st.session_state.selected_claim,
                            note_text,
                            "general",
                            "ai_agent"
                        )
                        if result.get('status') == 'success':
                            st.success("✅ Note added!")
                            st.json(result)
            
            with col3:
                task_desc = st.text_area("Task Description:", key="task_desc")
                assigned_to = st.text_input("Assign to:", key="assign_to")
                if st.button("📋 Create Task"):
                    if task_desc and assigned_to:
                        result = st.session_state.mcp_server.assign_investigation_task(
                            st.session_state.selected_claim,
                            task_desc,
                            assigned_to,
                            "high",
                            7
                        )
                        if result.get('status') == 'success':
                            st.success("✅ Task created!")
                            st.json(result)
    else:
        st.info("No claim selected. Go to Claim Search to select a claim.")
        if st.button("📋 Go to Claim Search"):
            st.session_state.selected_claim = None
            st.rerun()

# Investigation Notes Page
elif page == "📝 Investigation Notes":
    st.subheader("Investigation Notes & Tasks")
    
    tabs = st.tabs(["📝 Notes", "📋 Tasks", "📊 Status History"])
    
    with tabs[0]:
        # Load notes
        notes_query = f"""
        SELECT * FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.gold_investigation_notes
        WHERE is_deleted = FALSE
        ORDER BY created_timestamp DESC
        LIMIT 50
        """
        notes_results = st.session_state.mcp_server._execute_sql(notes_query)
        
        if notes_results and 'error' not in notes_results[0]:
            notes_df = pd.DataFrame(notes_results)
            st.dataframe(notes_df, use_container_width=True, hide_index=True)
        else:
            st.info("No investigation notes found.")
    
    with tabs[1]:
        # Load tasks
        tasks_query = f"""
        SELECT * FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.gold_investigation_tasks
        ORDER BY created_timestamp DESC
        LIMIT 50
        """
        tasks_results = st.session_state.mcp_server._execute_sql(tasks_query)
        
        if tasks_results and 'error' not in tasks_results[0]:
            tasks_df = pd.DataFrame(tasks_results)
            st.dataframe(tasks_df, use_container_width=True, hide_index=True)
        else:
            st.info("No investigation tasks found.")
    
    with tabs[2]:
        # Load status history
        audit_query = f"""
        SELECT * FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.gold_claim_status_audit
        ORDER BY update_timestamp DESC
        LIMIT 50
        """
        audit_results = st.session_state.mcp_server._execute_sql(audit_query)
        
        if audit_results and 'error' not in audit_results[0]:
            audit_df = pd.DataFrame(audit_results)
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
        else:
            st.info("No status audit records found.")

# Analytics Page
elif page == "📊 Analytics":
    st.subheader("Advanced Analytics & Insights")
    
    # Time series analysis
    st.subheader("📈 Claims Trend Over Time")
    time_series_query = f"""
    SELECT 
        DATE_TRUNC('month', incident_date) as month,
        COUNT(*) as claim_count,
        SUM(claim_amount) as total_amount,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count
    FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.silver_claims_enriched
    GROUP BY DATE_TRUNC('month', incident_date)
    ORDER BY month
    """
    ts_results = st.session_state.mcp_server._execute_sql(time_series_query)
    
    if ts_results and 'error' not in ts_results[0]:
        ts_df = pd.DataFrame(ts_results)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_df['month'],
            y=ts_df['claim_count'],
            mode='lines+markers',
            name='Total Claims',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=ts_df['month'],
            y=ts_df['fraud_count'],
            mode='lines+markers',
            name='Fraud Claims',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Claims Volume Trend',
            xaxis_title='Month',
            yaxis_title='Number of Claims',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    st.subheader("🎯 Fraud Detection Model Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Top risk factors
        risk_query = f"""
        SELECT 
            'Rapid Filing' as factor,
            SUM(CASE WHEN rapid_filing = 1 AND is_fraud THEN 1 ELSE 0 END) as fraud_count,
            SUM(CASE WHEN rapid_filing = 1 THEN 1 ELSE 0 END) as total_count
        FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.silver_fraud_features
        UNION ALL
        SELECT 
            'High Claim/Limit Ratio' as factor,
            SUM(CASE WHEN claim_near_limit = 1 AND is_fraud THEN 1 ELSE 0 END) as fraud_count,
            SUM(CASE WHEN claim_near_limit = 1 THEN 1 ELSE 0 END) as total_count
        FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.silver_fraud_features
        UNION ALL
        SELECT 
            'Prior Fraud' as factor,
            SUM(CASE WHEN has_prior_fraud = 1 AND is_fraud THEN 1 ELSE 0 END) as fraud_count,
            SUM(CASE WHEN has_prior_fraud = 1 THEN 1 ELSE 0 END) as total_count
        FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.silver_fraud_features
        """
        risk_results = st.session_state.mcp_server._execute_sql(risk_query)
        
        if risk_results and 'error' not in risk_results[0]:
            risk_df = pd.DataFrame(risk_results)
            risk_df['fraud_rate'] = (risk_df['fraud_count'] / risk_df['total_count'] * 100).round(1)
            
            fig = px.bar(
                risk_df,
                x='factor',
                y='fraud_rate',
                title='Fraud Rate by Risk Factor',
                labels={'fraud_rate': 'Fraud Rate (%)', 'factor': 'Risk Factor'},
                color='fraud_rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # State distribution
        state_query = f"""
        SELECT 
            address_state,
            COUNT(*) as claim_count,
            SUM(claim_amount) as total_amount
        FROM {st.session_state.mcp_server.catalog}.{st.session_state.mcp_server.schema}.silver_claims_enriched
        GROUP BY address_state
        ORDER BY claim_count DESC
        LIMIT 10
        """
        state_results = st.session_state.mcp_server._execute_sql(state_query)
        
        if state_results and 'error' not in state_results[0]:
            state_df = pd.DataFrame(state_results)
            
            fig = px.bar(
                state_df,
                x='address_state',
                y='claim_count',
                title='Top 10 States by Claim Volume',
                labels={'claim_count': 'Number of Claims', 'address_state': 'State'},
                color='total_amount',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.caption(
    "Databricks AI Insurance Claims Intelligence Platform | "
    "Powered by Databricks Lakehouse, MLflow, and Streamlit | "
    f"MCP Server: {len(st.session_state.mcp_server.list_tools()['tools'])} tools available"
)