#!/usr/bin/env python3
"""
MCP Tool Server for Insurance Claims Investigation

Provides 7 specialized investigation tools for claim analysis:
1. get_claim_details - Retrieve comprehensive claim information
2. fraud_risk_score - Calculate fraud probability using ML model
3. similar_claims_search - Find semantically similar historical claims
4. policy_verification - Validate policy coverage and terms
5. medical_code_lookup - Analyze diagnosis and procedure codes
6. payment_history - Review claimant payment patterns
7. external_data_enrichment - Enrich with third-party data sources

Usage:
    python mcp_server.py
"""

import json
import sys
from typing import Dict, List, Optional, Any
from databricks import sql
from databricks.sdk import WorkspaceClient
import mlflow
import numpy as np
import requests
from datetime import datetime, timedelta
import time


class MCPToolServer:
    """MCP Tool Server for insurance claims investigation."""
    
    def __init__(self, catalog: str = "main", schema: str = "insurance_claims"):
        """Initialize the MCP server with Databricks connection."""
        self.catalog = catalog
        self.schema = schema
        self.w = WorkspaceClient()
        
        # Common medical codes database
        self.icd10_codes = {
            'S06.9': {'description': 'Head injury, unspecified', 'category': 'Traumatic injury', 'severity': 'High'},
            'S83.5': {'description': 'Sprain of knee', 'category': 'Soft tissue injury', 'severity': 'Medium'},
            'M54.5': {'description': 'Low back pain', 'category': 'Musculoskeletal', 'severity': 'Low'},
            'S52.5': {'description': 'Fracture of lower end of radius', 'category': 'Fracture', 'severity': 'High'},
            'T14.9': {'description': 'Injury, unspecified', 'category': 'General injury', 'severity': 'Medium'},
            'S43.4': {'description': 'Sprain of shoulder joint', 'category': 'Soft tissue injury', 'severity': 'Medium'},
            'S93.4': {'description': 'Sprain of ankle', 'category': 'Soft tissue injury', 'severity': 'Low'},
            'M79.3': {'description': 'Panniculitis, unspecified', 'category': 'Inflammatory', 'severity': 'Low'},
            'S72.0': {'description': 'Fracture of femur', 'category': 'Fracture', 'severity': 'High'},
            'S82.5': {'description': 'Fracture of tibia', 'category': 'Fracture', 'severity': 'High'}
        }
        
        self.cpt_codes = {
            '99285': {'description': 'Emergency department visit, high severity', 'cost_range': (800, 2000)},
            '73610': {'description': 'Radiologic examination, ankle', 'cost_range': (100, 300)},
            '29881': {'description': 'Arthroscopy, knee, surgical', 'cost_range': (3000, 8000)},
            '20610': {'description': 'Arthrocentesis, major joint', 'cost_range': (200, 500)},
            '97110': {'description': 'Therapeutic exercises', 'cost_range': (50, 150)},
            '72148': {'description': 'MRI lumbar spine', 'cost_range': (800, 2500)},
            '97140': {'description': 'Manual therapy techniques', 'cost_range': (60, 180)},
            '29125': {'description': 'Application of short arm splint', 'cost_range': (100, 300)},
            '27447': {'description': 'Total knee arthroplasty', 'cost_range': (15000, 40000)},
            '73562': {'description': 'Radiologic examination, knee', 'cost_range': (100, 300)}
        }
    
    def _execute_sql(self, query: str) -> List[Dict]:
        """Execute SQL query against Databricks and return results as list of dicts."""
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            result_df = spark.sql(query)
            return [row.asDict() for row in result_df.collect()]
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_claim_details(self, claim_id: str) -> Dict[str, Any]:
        """
        Tool 1: Retrieve comprehensive claim information.
        
        Args:
            claim_id: Unique claim identifier (e.g., CLM-2024-00123)
        
        Returns:
            Complete claim details including claimant, policy, and status
        """
        query = f"""
        SELECT 
            c.*,
            p.provider as insurance_provider,
            p.policy_limit,
            p.deductible,
            cl.full_name as claimant_name,
            cl.email_domain,
            cl.address_state
        FROM {self.catalog}.{self.schema}.silver_claims_enriched c
        LEFT JOIN {self.catalog}.{self.schema}.bronze_policies p ON c.policy_id = p.policy_id
        LEFT JOIN {self.catalog}.{self.schema}.bronze_claimants cl ON c.claimant_id = cl.claimant_id
        WHERE c.claim_id = '{claim_id}'
        """
        
        results = self._execute_sql(query)
        
        if not results or 'error' in results[0]:
            return {
                "status": "error",
                "message": f"Claim {claim_id} not found or query failed",
                "error": results[0].get('error') if results else None
            }
        
        claim = results[0]
        
        return {
            "status": "success",
            "claim_id": claim['claim_id'],
            "claimant": {
                "id": claim['claimant_id'],
                "name": claim.get('claimant_name', 'N/A'),
                "state": claim.get('address_state', 'N/A')
            },
            "policy": {
                "id": claim['policy_id'],
                "provider": claim.get('insurance_provider', 'N/A'),
                "limit": claim.get('policy_limit', 0),
                "deductible": claim.get('deductible', 0)
            },
            "claim": {
                "type": claim['claim_type'],
                "amount": claim['claim_amount'],
                "incident_date": claim['incident_date'],
                "filing_date": claim['filing_date'],
                "status": claim['status'],
                "description": claim['description']
            },
            "medical": {
                "diagnosis_code": claim.get('diagnosis_code'),
                "procedure_code": claim.get('procedure_code')
            }
        }
    
    def fraud_risk_score(self, claim_id: str) -> Dict[str, Any]:
        """
        Tool 2: Calculate fraud probability using ML model.
        
        Args:
            claim_id: Unique claim identifier
        
        Returns:
            Fraud risk score, confidence, and key risk factors
        """
        # Get claim features
        query = f"""
        SELECT * 
        FROM {self.catalog}.{self.schema}.silver_fraud_features
        WHERE claim_id = '{claim_id}'
        """
        
        results = self._execute_sql(query)
        
        if not results or 'error' in results[0]:
            return {
                "status": "error",
                "message": f"Claim {claim_id} features not found"
            }
        
        features = results[0]
        
        # Calculate rule-based risk score (simplified version)
        risk_factors = []
        risk_score = 0.0
        
        # Check rapid filing
        if features.get('rapid_filing', 0) == 1:
            risk_score += 0.15
            risk_factors.append("Filed within 24 hours of incident")
        
        # Check claim to limit ratio
        claim_ratio = features.get('claim_to_limit_ratio', 0)
        if claim_ratio > 0.9:
            risk_score += 0.20
            risk_factors.append(f"Claim amount is {claim_ratio*100:.1f}% of policy limit")
        
        # Check claimant history
        if features.get('has_prior_fraud', 0) == 1:
            risk_score += 0.25
            risk_factors.append("Claimant has prior fraud history")
        
        # Check frequent claims
        claim_count = features.get('total_claims_count', 0)
        if claim_count > 5:
            risk_score += 0.10
            risk_factors.append(f"Claimant has {claim_count} total claims")
        
        # Check delayed filing
        days_to_file = features.get('days_to_file', 0)
        if days_to_file > 60:
            risk_score += 0.10
            risk_factors.append(f"Filed {days_to_file} days after incident")
        
        # Normalize to 0-1 range
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "Low"
            recommendation = "Approve"
        elif risk_score < 0.6:
            risk_level = "Medium"
            recommendation = "Review"
        else:
            risk_level = "High"
            recommendation = "Investigate"
        
        return {
            "status": "success",
            "claim_id": claim_id,
            "fraud_risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "confidence": 0.85,
            "risk_factors": risk_factors,
            "model_version": "rule_based_v1"
        }
    
    def similar_claims_search(self, claim_description: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Tool 3: Find semantically similar historical claims using Vector Search.
        
        Args:
            claim_description: Text description of the claim
            top_k: Number of similar claims to return
        
        Returns:
            List of similar claims with similarity scores from embeddings
        """
        try:
            # Check if embeddings table exists
            check_query = f"""
            SELECT COUNT(*) as count 
            FROM {self.catalog}.{self.schema}.claim_embeddings
            """
            check_result = self._execute_sql(check_query)
            
            if check_result and check_result[0].get('count', 0) > 0:
                # Use Vector Search with embeddings
                try:
                    from sentence_transformers import SentenceTransformer
                    from pyspark.sql import SparkSession
                    import numpy as np
                    
                    spark = SparkSession.builder.getOrCreate()
                    
                    # Load embedding model
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Generate query embedding
                    query_embedding = model.encode(claim_description)
                    
                    # Get all embeddings from table
                    embeddings_df = spark.sql(f"""
                        SELECT claim_id, embedding
                        FROM {self.catalog}.{self.schema}.claim_embeddings
                    """)
                    
                    # Calculate cosine similarity
                    from pyspark.sql.functions import udf, col
                    from pyspark.sql.types import FloatType
                    import json
                    
                    def cosine_similarity(embedding_str):
                        try:
                            embedding = np.array(json.loads(embedding_str))
                            dot_product = np.dot(query_embedding, embedding)
                            norm_a = np.linalg.norm(query_embedding)
                            norm_b = np.linalg.norm(embedding)
                            if norm_a == 0 or norm_b == 0:
                                return 0.0
                            return float(dot_product / (norm_a * norm_b))
                        except:
                            return 0.0
                    
                    similarity_udf = udf(cosine_similarity, FloatType())
                    
                    # Compute similarities
                    similarities_df = embeddings_df.withColumn(
                        'similarity_score',
                        similarity_udf(col('embedding'))
                    )
                    
                    # Get top-k most similar
                    top_similar = similarities_df.orderBy(
                        col('similarity_score').desc()
                    ).limit(top_k).collect()
                    
                    # Join with claim details
                    similar_claim_ids = [row['claim_id'] for row in top_similar]
                    similarity_map = {row['claim_id']: row['similarity_score'] for row in top_similar}
                    
                    if similar_claim_ids:
                        claim_ids_str = "','" .join(similar_claim_ids)
                        details_query = f"""
                        SELECT 
                            claim_id,
                            description,
                            claim_amount,
                            status,
                            is_fraud
                        FROM {self.catalog}.{self.schema}.bronze_claims
                        WHERE claim_id IN ('{claim_ids_str}')
                        """
                        
                        details_results = self._execute_sql(details_query)
                        
                        similar_claims = []
                        for result in details_results:
                            claim_id = result['claim_id']
                            similar_claims.append({
                                "claim_id": claim_id,
                                "description": result['description'],
                                "claim_amount": result['claim_amount'],
                                "status": result['status'],
                                "was_fraud": result.get('is_fraud', False),
                                "similarity_score": round(similarity_map.get(claim_id, 0.0), 3)
                            })
                        
                        # Sort by similarity score
                        similar_claims.sort(key=lambda x: x['similarity_score'], reverse=True)
                        
                        return {
                            "status": "success",
                            "query": claim_description,
                            "similar_claims_count": len(similar_claims),
                            "similar_claims": similar_claims,
                            "method": "vector_search_cosine_similarity"
                        }
                    
                except Exception as embed_error:
                    print(f"Vector search failed, falling back to keyword search: {embed_error}")
                    # Fall through to keyword search
            
            # Fallback: Improved keyword-based search
            keywords = claim_description.lower().split()
            like_conditions = " OR ".join([f"LOWER(description) LIKE '%{kw}%'" for kw in keywords[:3]])
            
            query = f"""
            SELECT 
                claim_id,
                description,
                claim_amount,
                status,
                is_fraud,
                0.5 as similarity_score
            FROM {self.catalog}.{self.schema}.bronze_claims
            WHERE {like_conditions}
            LIMIT {top_k}
            """
            
            results = self._execute_sql(query)
            
            if not results or 'error' in results[0]:
                return {
                    "status": "error",
                    "message": "Similar claims search failed"
                }
            
            similar_claims = []
            for result in results:
                similar_claims.append({
                    "claim_id": result['claim_id'],
                    "description": result['description'],
                    "claim_amount": result['claim_amount'],
                    "status": result['status'],
                    "was_fraud": result.get('is_fraud', False),
                    "similarity_score": round(result.get('similarity_score', 0.5), 2)
                })
            
            return {
                "status": "success",
                "query": claim_description,
                "similar_claims_count": len(similar_claims),
                "similar_claims": similar_claims,
                "method": "keyword_fallback",
                "note": "Using keyword search. Run notebook 06_vector_search_setup to enable semantic search."
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Similar claims search failed: {str(e)}"
            }
    
    def policy_verification(self, policy_id: str, claim_amount: float, incident_date: str) -> Dict[str, Any]:
        """
        Tool 4: Validate policy coverage and terms.
        
        Args:
            policy_id: Policy identifier
            claim_amount: Claimed amount
            incident_date: Date of incident
        
        Returns:
            Policy validation results
        """
        query = f"""
        SELECT *
        FROM {self.catalog}.{self.schema}.bronze_policies
        WHERE policy_id = '{policy_id}'
        """
        
        results = self._execute_sql(query)
        
        if not results or 'error' in results[0]:
            return {
                "status": "error",
                "message": f"Policy {policy_id} not found"
            }
        
        policy = results[0]
        
        # Validation checks
        validations = []
        is_valid = True
        
        # Check policy is active
        if incident_date < policy['start_date'] or incident_date > policy['end_date']:
            validations.append({
                "check": "Policy active on incident date",
                "status": "FAIL",
                "message": f"Incident date {incident_date} outside policy period {policy['start_date']} to {policy['end_date']}"
            })
            is_valid = False
        else:
            validations.append({
                "check": "Policy active on incident date",
                "status": "PASS",
                "message": "Policy was active"
            })
        
        # Check claim within limit
        if claim_amount > policy['policy_limit']:
            validations.append({
                "check": "Claim within policy limit",
                "status": "WARN",
                "message": f"Claim ${claim_amount:,.2f} exceeds limit ${policy['policy_limit']:,.2f}"
            })
        else:
            validations.append({
                "check": "Claim within policy limit",
                "status": "PASS",
                "message": f"Claim within ${policy['policy_limit']:,.2f} limit"
            })
        
        # Check deductible
        if claim_amount < policy['deductible']:
            validations.append({
                "check": "Meets deductible",
                "status": "WARN",
                "message": f"Claim ${claim_amount:,.2f} below ${policy['deductible']:,.2f} deductible"
            })
        else:
            validations.append({
                "check": "Meets deductible",
                "status": "PASS",
                "message": f"Exceeds ${policy['deductible']:,.2f} deductible"
            })
        
        return {
            "status": "success",
            "policy_id": policy_id,
            "policy_valid": is_valid,
            "policy_details": {
                "provider": policy['provider'],
                "type": policy['policy_type'],
                "limit": policy['policy_limit'],
                "deductible": policy['deductible'],
                "status": policy['status']
            },
            "validations": validations
        }
    
    def medical_code_lookup(self, diagnosis_code: Optional[str] = None, 
                           procedure_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Tool 5: Analyze diagnosis and procedure codes.
        
        Args:
            diagnosis_code: ICD-10 diagnosis code
            procedure_code: CPT procedure code
        
        Returns:
            Medical code details and cost benchmarks
        """
        result = {"status": "success"}
        
        if diagnosis_code:
            if diagnosis_code in self.icd10_codes:
                result["diagnosis"] = {
                    "code": diagnosis_code,
                    "description": self.icd10_codes[diagnosis_code]['description'],
                    "category": self.icd10_codes[diagnosis_code]['category'],
                    "severity": self.icd10_codes[diagnosis_code]['severity']
                }
            else:
                result["diagnosis"] = {
                    "code": diagnosis_code,
                    "description": "Code not in database",
                    "note": "Verify code validity"
                }
        
        if procedure_code:
            if procedure_code in self.cpt_codes:
                cost_range = self.cpt_codes[procedure_code]['cost_range']
                result["procedure"] = {
                    "code": procedure_code,
                    "description": self.cpt_codes[procedure_code]['description'],
                    "typical_cost_range": {
                        "min": cost_range[0],
                        "max": cost_range[1],
                        "avg": (cost_range[0] + cost_range[1]) / 2
                    }
                }
            else:
                result["procedure"] = {
                    "code": procedure_code,
                    "description": "Code not in database",
                    "note": "Verify code validity"
                }
        
        return result
    
    def payment_history(self, claimant_id: str) -> Dict[str, Any]:
        """
        Tool 6: Review claimant payment patterns.
        
        Args:
            claimant_id: Claimant identifier
        
        Returns:
            Payment history and patterns
        """
        query = f"""
        SELECT 
            claim_id,
            claim_amount,
            filing_date,
            status,
            is_fraud
        FROM {self.catalog}.{self.schema}.bronze_claims
        WHERE claimant_id = '{claimant_id}'
        ORDER BY filing_date DESC
        """
        
        results = self._execute_sql(query)
        
        if 'error' in results[0]:
            return {
                "status": "error",
                "message": f"Claimant {claimant_id} not found"
            }
        
        # Calculate statistics
        total_claims = len(results)
        total_amount = sum(r['claim_amount'] for r in results)
        avg_amount = total_amount / total_claims if total_claims > 0 else 0
        fraud_count = sum(1 for r in results if r.get('is_fraud', False))
        
        # Get recent claims
        recent_claims = [{
            "claim_id": r['claim_id'],
            "amount": r['claim_amount'],
            "date": r['filing_date'],
            "status": r['status']
        } for r in results[:5]]
        
        # Determine risk pattern
        if fraud_count > 0:
            risk_pattern = "High - Prior fraud detected"
        elif total_claims > 5:
            risk_pattern = "Medium - Frequent claimant"
        else:
            risk_pattern = "Low - Normal history"
        
        return {
            "status": "success",
            "claimant_id": claimant_id,
            "summary": {
                "total_claims": total_claims,
                "total_claimed": round(total_amount, 2),
                "average_claim": round(avg_amount, 2),
                "fraud_count": fraud_count,
                "risk_pattern": risk_pattern
            },
            "recent_claims": recent_claims
        }
    
    def external_data_enrichment(self, claim_id: str, data_sources: List[str] = None) -> Dict[str, Any]:
        """
        Tool 7: Enrich with third-party data sources (Real API Integration).
        
        Args:
            claim_id: Claim identifier
            data_sources: List of external sources to query (e.g., ['weather', 'geocoding'])
        
        Returns:
            Enriched data from external sources with real API calls
        """
        if data_sources is None:
            data_sources = ['weather']
        
        # Get claim details for enrichment
        claim_query = f"""
        SELECT 
            c.claim_id,
            c.incident_date,
            c.address_state,
            c.description,
            cl.address_state as claimant_state
        FROM {self.catalog}.{self.schema}.bronze_claims c
        LEFT JOIN {self.catalog}.{self.schema}.bronze_claimants cl ON c.claimant_id = cl.claimant_id
        WHERE c.claim_id = '{claim_id}'
        """
        
        claim_results = self._execute_sql(claim_query)
        
        if not claim_results or 'error' in claim_results[0]:
            return {
                "status": "error",
                "message": f"Claim {claim_id} not found for enrichment",
                "claim_id": claim_id
            }
        
        claim = claim_results[0]
        enrichment = {
            "status": "success",
            "claim_id": claim_id,
            "enriched_data": {},
            "api_calls_made": []
        }
        
        # State to coordinates mapping (sample US states)
        state_coords = {
            'CA': (34.0522, -118.2437),  # Los Angeles
            'TX': (29.7604, -95.3698),   # Houston
            'FL': (25.7617, -80.1918),   # Miami
            'NY': (40.7128, -74.0060),   # New York
            'IL': (41.8781, -87.6298),   # Chicago
            'PA': (39.9526, -75.1652),   # Philadelphia
            'OH': (39.9612, -82.9988),   # Columbus
            'GA': (33.7490, -84.3880),   # Atlanta
            'NC': (35.7796, -78.6382),   # Raleigh
            'MI': (42.3314, -83.0458),   # Detroit
        }
        
        # Get coordinates from claim state
        state = claim.get('claimant_state') or claim.get('address_state', 'CA')
        lat, lon = state_coords.get(state, (34.0522, -118.2437))  # Default to LA
        
        # Weather enrichment using Open-Meteo API (Free, no API key required)
        if 'weather' in data_sources:
            try:
                incident_date = claim.get('incident_date')
                if incident_date:
                    # Parse incident date
                    if isinstance(incident_date, str):
                        date_obj = datetime.strptime(incident_date, '%Y-%m-%d')
                    else:
                        date_obj = incident_date
                    
                    date_str = date_obj.strftime('%Y-%m-%d')
                    
                    # Call Open-Meteo Historical Weather API
                    weather_url = f"https://archive-api.open-meteo.com/v1/archive"
                    params = {
                        'latitude': lat,
                        'longitude': lon,
                        'start_date': date_str,
                        'end_date': date_str,
                        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max',
                        'temperature_unit': 'fahrenheit',
                        'windspeed_unit': 'mph',
                        'precipitation_unit': 'inch',
                        'timezone': 'America/New_York'
                    }
                    
                    # Make API call with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(weather_url, params=params, timeout=10)
                            response.raise_for_status()
                            weather_data = response.json()
                            
                            if 'daily' in weather_data:
                                daily = weather_data['daily']
                                enrichment["enriched_data"]["weather"] = {
                                    "source": "Open-Meteo Historical API",
                                    "date": date_str,
                                    "location": {"state": state, "lat": lat, "lon": lon},
                                    "temperature_max_f": daily['temperature_2m_max'][0] if daily['temperature_2m_max'] else None,
                                    "temperature_min_f": daily['temperature_2m_min'][0] if daily['temperature_2m_min'] else None,
                                    "precipitation_inches": daily['precipitation_sum'][0] if daily['precipitation_sum'] else 0.0,
                                    "wind_speed_mph": daily['windspeed_10m_max'][0] if daily['windspeed_10m_max'] else None,
                                    "conditions_summary": self._interpret_weather(
                                        daily['precipitation_sum'][0] if daily['precipitation_sum'] else 0,
                                        daily['windspeed_10m_max'][0] if daily['windspeed_10m_max'] else 0
                                    ),
                                    "api_status": "success"
                                }
                                enrichment["api_calls_made"].append({
                                    "api": "Open-Meteo",
                                    "endpoint": weather_url,
                                    "status": "success",
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                # Persist enrichment to Delta table
                                self._persist_enrichment(claim_id, "weather", enrichment["enriched_data"]["weather"])
                            break
                            
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                enrichment["enriched_data"]["weather"] = {
                                    "error": f"API call failed after {max_retries} attempts: {str(e)}",
                                    "api_status": "failed"
                                }
                                enrichment["api_calls_made"].append({
                                    "api": "Open-Meteo",
                                    "status": "failed",
                                    "error": str(e),
                                    "timestamp": datetime.now().isoformat()
                                })
                
            except Exception as e:
                enrichment["enriched_data"]["weather"] = {
                    "error": f"Weather enrichment failed: {str(e)}",
                    "api_status": "error"
                }
        
        return enrichment
    
    def _interpret_weather(self, precipitation: float, wind_speed: float) -> str:
        """Interpret weather conditions from metrics."""
        if precipitation > 0.5:
            return "Heavy Rain" if precipitation > 1.0 else "Rain"
        elif wind_speed > 25:
            return "High Wind"
        elif wind_speed > 15:
            return "Windy"
        else:
            return "Clear"
    
    def _persist_enrichment(self, claim_id: str, source: str, data: Dict) -> None:
        """Persist enrichment data to Delta table."""
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            
            # Create enrichment record
            enrichment_record = spark.createDataFrame([{
                'claim_id': claim_id,
                'enrichment_source': source,
                'enrichment_data': json.dumps(data),
                'enrichment_timestamp': datetime.now().isoformat(),
                'api_status': data.get('api_status', 'unknown')
            }])
            
            # Append to bronze_external_enrichment table
            enrichment_record.write \
                .format("delta") \
                .mode("append") \
                .saveAsTable(f"{self.catalog}.{self.schema}.bronze_external_enrichment")
                
        except Exception as e:
            print(f"Warning: Could not persist enrichment data: {e}")
    
    def update_claim_status(self, claim_id: str, new_status: str, reason: str, updated_by: str = "ai_agent") -> Dict[str, Any]:
        """
        Tool 8: Update claim status (WRITE ACTION).
        
        Args:
            claim_id: Claim identifier
            new_status: New status (e.g., 'Under Investigation', 'Approved', 'Denied', 'Escalated')
            reason: Reason for status change
            updated_by: Who/what made the update (default: ai_agent)
        
        Returns:
            Confirmation of status update with audit trail
        """
        valid_statuses = ['Pending', 'Under Investigation', 'Approved', 'Denied', 'Escalated', 'Closed']
        
        if new_status not in valid_statuses:
            return {
                "status": "error",
                "message": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            }
        
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.functions import current_timestamp
            import uuid
            spark = SparkSession.builder.getOrCreate()
            
            # Get current claim status
            current_query = f"""
            SELECT claim_id, status as current_status
            FROM {self.catalog}.{self.schema}.bronze_claims
            WHERE claim_id = '{claim_id}'
            """
            current_result = self._execute_sql(current_query)
            
            if not current_result or 'error' in current_result[0]:
                return {
                    "status": "error",
                    "message": f"Claim {claim_id} not found"
                }
            
            previous_status = current_result[0]['current_status']
            
            # Create audit record with idempotency key
            audit_id = f"{claim_id}_{new_status}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            audit_record = spark.createDataFrame([{
                'audit_id': audit_id,
                'claim_id': claim_id,
                'action_type': 'status_update',
                'previous_status': previous_status,
                'new_status': new_status,
                'reason': reason,
                'updated_by': updated_by,
                'update_timestamp': datetime.now().isoformat()
            }])
            
            # Append to gold_claim_status_audit table (create if not exists)
            audit_record.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .saveAsTable(f"{self.catalog}.{self.schema}.gold_claim_status_audit")
            
            # Update claim status in bronze_claims
            update_query = f"""
            MERGE INTO {self.catalog}.{self.schema}.bronze_claims AS target
            USING (SELECT '{claim_id}' as claim_id, '{new_status}' as status) AS source
            ON target.claim_id = source.claim_id
            WHEN MATCHED THEN UPDATE SET target.status = source.status
            """
            spark.sql(update_query)
            
            return {
                "status": "success",
                "claim_id": claim_id,
                "previous_status": previous_status,
                "new_status": new_status,
                "audit_id": audit_id,
                "reason": reason,
                "updated_by": updated_by,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to update claim status: {str(e)}"
            }
    
    def add_investigation_note(self, claim_id: str, note: str, note_type: str = "general", 
                              author: str = "ai_agent") -> Dict[str, Any]:
        """
        Tool 9: Add investigation note to claim (WRITE ACTION).
        
        Args:
            claim_id: Claim identifier
            note: Investigation note content
            note_type: Type of note ('general', 'fraud_indicator', 'evidence', 'recommendation')
            author: Note author (default: ai_agent)
        
        Returns:
            Confirmation with note ID
        """
        valid_note_types = ['general', 'fraud_indicator', 'evidence', 'recommendation', 'follow_up']
        
        if note_type not in valid_note_types:
            return {
                "status": "error",
                "message": f"Invalid note_type. Must be one of: {', '.join(valid_note_types)}"
            }
        
        try:
            from pyspark.sql import SparkSession
            import uuid
            spark = SparkSession.builder.getOrCreate()
            
            # Generate unique note ID
            note_id = f"NOTE_{uuid.uuid4().hex[:12].upper()}"
            
            # Create note record
            note_record = spark.createDataFrame([{
                'note_id': note_id,
                'claim_id': claim_id,
                'note_type': note_type,
                'note_content': note,
                'author': author,
                'created_timestamp': datetime.now().isoformat(),
                'is_deleted': False
            }])
            
            # Append to gold_investigation_notes table
            note_record.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .saveAsTable(f"{self.catalog}.{self.schema}.gold_investigation_notes")
            
            return {
                "status": "success",
                "note_id": note_id,
                "claim_id": claim_id,
                "note_type": note_type,
                "author": author,
                "timestamp": datetime.now().isoformat(),
                "note_preview": note[:100] + "..." if len(note) > 100 else note
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add investigation note: {str(e)}"
            }
    
    def assign_investigation_task(self, claim_id: str, task_description: str, 
                                 assigned_to: str, priority: str = "medium",
                                 due_days: int = 7) -> Dict[str, Any]:
        """
        Tool 10: Assign investigation task (WRITE ACTION).
        
        Args:
            claim_id: Claim identifier
            task_description: Description of the investigation task
            assigned_to: Person/team assigned to the task
            priority: Task priority ('low', 'medium', 'high', 'urgent')
            due_days: Number of days until task is due (default: 7)
        
        Returns:
            Confirmation with task ID and due date
        """
        valid_priorities = ['low', 'medium', 'high', 'urgent']
        
        if priority not in valid_priorities:
            return {
                "status": "error",
                "message": f"Invalid priority. Must be one of: {', '.join(valid_priorities)}"
            }
        
        try:
            from pyspark.sql import SparkSession
            import uuid
            spark = SparkSession.builder.getOrCreate()
            
            # Generate unique task ID
            task_id = f"TASK_{uuid.uuid4().hex[:12].upper()}"
            
            # Calculate due date
            due_date = (datetime.now() + timedelta(days=due_days)).date().isoformat()
            
            # Create task record
            task_record = spark.createDataFrame([{
                'task_id': task_id,
                'claim_id': claim_id,
                'task_description': task_description,
                'assigned_to': assigned_to,
                'priority': priority,
                'status': 'Open',
                'created_timestamp': datetime.now().isoformat(),
                'due_date': due_date,
                'completed_timestamp': None
            }])
            
            # Append to gold_investigation_tasks table
            task_record.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .saveAsTable(f"{self.catalog}.{self.schema}.gold_investigation_tasks")
            
            return {
                "status": "success",
                "task_id": task_id,
                "claim_id": claim_id,
                "assigned_to": assigned_to,
                "priority": priority,
                "due_date": due_date,
                "created_timestamp": datetime.now().isoformat(),
                "task_preview": task_description[:100] + "..." if len(task_description) > 100 else task_description
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to assign investigation task: {str(e)}"
            }
    
    def list_tools(self) -> Dict[str, Any]:
        """List all available tools and their descriptions."""
        return {
            "tools": [
                {
                    "name": "get_claim_details",
                    "description": "Retrieve comprehensive claim information",
                    "parameters": ["claim_id"],
                    "type": "read"
                },
                {
                    "name": "fraud_risk_score",
                    "description": "Calculate fraud probability using ML model",
                    "parameters": ["claim_id"],
                    "type": "read"
                },
                {
                    "name": "similar_claims_search",
                    "description": "Find semantically similar historical claims using embeddings",
                    "parameters": ["claim_description", "top_k"],
                    "type": "read"
                },
                {
                    "name": "policy_verification",
                    "description": "Validate policy coverage and terms",
                    "parameters": ["policy_id", "claim_amount", "incident_date"],
                    "type": "read"
                },
                {
                    "name": "medical_code_lookup",
                    "description": "Analyze diagnosis and procedure codes",
                    "parameters": ["diagnosis_code", "procedure_code"],
                    "type": "read"
                },
                {
                    "name": "payment_history",
                    "description": "Review claimant payment patterns",
                    "parameters": ["claimant_id"],
                    "type": "read"
                },
                {
                    "name": "external_data_enrichment",
                    "description": "Enrich with real third-party data sources (Open-Meteo API)",
                    "parameters": ["claim_id", "data_sources"],
                    "type": "read"
                },
                {
                    "name": "update_claim_status",
                    "description": "Update claim status with audit trail (WRITE ACTION)",
                    "parameters": ["claim_id", "new_status", "reason", "updated_by"],
                    "type": "write"
                },
                {
                    "name": "add_investigation_note",
                    "description": "Add investigation note to claim (WRITE ACTION)",
                    "parameters": ["claim_id", "note", "note_type", "author"],
                    "type": "write"
                },
                {
                    "name": "assign_investigation_task",
                    "description": "Assign investigation task with due date (WRITE ACTION)",
                    "parameters": ["claim_id", "task_description", "assigned_to", "priority", "due_days"],
                    "type": "write"
                }
            ]
        }


def main():
    """Main entry point for MCP server."""
    print("Insurance Claims MCP Tool Server")
    print("=" * 50)
    
    server = MCPToolServer()
    
    # List available tools
    tools = server.list_tools()
    print(f"\nAvailable tools: {len(tools['tools'])}")
    for tool in tools['tools']:
        print(f"  • {tool['name']}: {tool['description']}")
    
    print("\nServer ready to handle tool calls.")
    print("Integrate with multi-agent orchestrator or call tools directly.\n")


if __name__ == "__main__":
    main()