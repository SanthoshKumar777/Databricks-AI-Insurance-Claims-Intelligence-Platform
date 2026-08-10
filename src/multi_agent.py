#!/usr/bin/env python3
"""
Multi-Agent Orchestration System for Insurance Claims

Central router with 4 specialized agents:
1. Fraud Detection Agent - ML-based fraud scoring and pattern recognition
2. Legal Compliance Agent - Regulatory compliance and policy interpretation  
3. Medical Review Agent - Clinical documentation and medical necessity
4. Financial Analysis Agent - Cost analysis and reserve recommendations

Architecture:
- Central Router analyzes claim and routes to appropriate specialists
- Each agent has access to MCP tools for investigation
- Router aggregates specialist responses into unified recommendation
- All decisions logged to Lakehouse for audit and evaluation
"""

import json
import sys
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit

# Import MCP server
from mcp_server import MCPToolServer


class BaseAgent:
    """Base class for all specialist agents."""
    
    def __init__(self, name: str, mcp_server: MCPToolServer):
        self.name = name
        self.mcp = mcp_server
        self.w = WorkspaceClient()
    
    def analyze(self, claim_data: Dict) -> Dict[str, Any]:
        """Override in subclass to implement specialist logic."""
        raise NotImplementedError


class FraudDetectionAgent(BaseAgent):
    """Agent specialized in fraud detection and pattern recognition."""
    
    def __init__(self, mcp_server: MCPToolServer):
        super().__init__("Fraud Detection Agent", mcp_server)
    
    def analyze(self, claim_data: Dict) -> Dict[str, Any]:
        """
        Analyze claim for fraud indicators using ML model and MCP tools.
        """
        claim_id = claim_data['claim_id']
        
        # Get fraud risk score from MCP tool
        fraud_analysis = self.mcp.fraud_risk_score(claim_id)
        
        # Search for similar historical claims
        similar_claims = self.mcp.similar_claims_search(
            claim_data.get('description', ''),
            top_k=5
        )
        
        # Check claimant history
        payment_history = self.mcp.payment_history(
            claim_data['claimant_id']
        )
        
        # Aggregate insights
        risk_score = fraud_analysis.get('fraud_risk_score', 0.0)
        risk_level = fraud_analysis.get('risk_level', 'Unknown')
        risk_factors = fraud_analysis.get('risk_factors', [])
        
        # Count fraud in similar claims
        similar_fraud_count = sum(
            1 for c in similar_claims.get('similar_claims', [])
            if c.get('was_fraud', False)
        )
        
        # Compile recommendation
        if risk_score > 0.7:
            recommendation = "DENY"
            confidence = 0.9
            rationale = f"High fraud risk (score: {risk_score:.2f}). {', '.join(risk_factors[:3])}"
        elif risk_score > 0.5:
            recommendation = "INVESTIGATE"
            confidence = 0.75
            rationale = f"Medium fraud risk (score: {risk_score:.2f}). Further investigation recommended."
        else:
            recommendation = "APPROVE"
            confidence = 0.85
            rationale = f"Low fraud risk (score: {risk_score:.2f}). Claim appears legitimate."
        
        return {
            "agent": self.name,
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "analysis": {
                "fraud_risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "similar_fraud_count": similar_fraud_count,
                "total_similar_claims": similar_claims.get('similar_claims_count', 0),
                "claimant_history": payment_history.get('summary', {})
            }
        }


class LegalComplianceAgent(BaseAgent):
    """Agent specialized in legal compliance and policy interpretation."""
    
    def __init__(self, mcp_server: MCPToolServer):
        super().__init__("Legal Compliance Agent", mcp_server)
    
    def analyze(self, claim_data: Dict) -> Dict[str, Any]:
        """
        Verify policy compliance and regulatory requirements.
        """
        # Verify policy coverage
        policy_check = self.mcp.policy_verification(
            claim_data['policy_id'],
            claim_data['claim_amount'],
            claim_data['incident_date']
        )
        
        policy_valid = policy_check.get('policy_valid', False)
        validations = policy_check.get('validations', [])
        
        # Assess compliance
        failed_checks = [v for v in validations if v['status'] == 'FAIL']
        warning_checks = [v for v in validations if v['status'] == 'WARN']
        
        if failed_checks:
            recommendation = "DENY"
            confidence = 0.95
            rationale = f"Policy validation failed: {failed_checks[0]['message']}"
        elif warning_checks:
            recommendation = "REVIEW"
            confidence = 0.7
            rationale = f"Policy warnings detected: {warning_checks[0]['message']}"
        else:
            recommendation = "APPROVE"
            confidence = 0.9
            rationale = "All policy validations passed. Claim compliant with policy terms."
        
        return {
            "agent": self.name,
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "analysis": {
                "policy_valid": policy_valid,
                "validations": validations,
                "failed_checks": len(failed_checks),
                "warning_checks": len(warning_checks)
            }
        }


class MedicalReviewAgent(BaseAgent):
    """Agent specialized in medical review and clinical analysis."""
    
    def __init__(self, mcp_server: MCPToolServer):
        super().__init__("Medical Review Agent", mcp_server)
    
    def analyze(self, claim_data: Dict) -> Dict[str, Any]:
        """
        Review medical codes and clinical documentation.
        """
        # Skip if not a medical claim
        if claim_data['claim_type'] not in ['Health', 'Workers Comp']:
            return {
                "agent": self.name,
                "recommendation": "N/A",
                "confidence": 1.0,
                "rationale": "Not a medical claim - no medical review required.",
                "analysis": {}
            }
        
        # Lookup medical codes
        medical_lookup = self.mcp.medical_code_lookup(
            diagnosis_code=claim_data.get('diagnosis_code'),
            procedure_code=claim_data.get('procedure_code')
        )
        
        # Analyze medical appropriateness
        diagnosis = medical_lookup.get('diagnosis', {})
        procedure = medical_lookup.get('procedure', {})
        
        # Check if claim amount aligns with typical procedure cost
        if procedure.get('typical_cost_range'):
            cost_range = procedure['typical_cost_range']
            claim_amount = claim_data['claim_amount']
            
            if claim_amount > cost_range['max'] * 1.5:
                recommendation = "INVESTIGATE"
                confidence = 0.7
                rationale = f"Claim amount ${claim_amount:,.2f} significantly exceeds typical range ${cost_range['min']:,.2f}-${cost_range['max']:,.2f}"
            elif claim_amount < cost_range['min'] * 0.5:
                recommendation = "REVIEW"
                confidence = 0.6
                rationale = f"Claim amount unusually low for procedure {procedure.get('code', 'N/A')}"
            else:
                recommendation = "APPROVE"
                confidence = 0.85
                rationale = "Medical codes valid and claim amount within expected range."
        else:
            recommendation = "REVIEW"
            confidence = 0.5
            rationale = "Medical codes not in database - manual review recommended."
        
        return {
            "agent": self.name,
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "analysis": {
                "diagnosis": diagnosis,
                "procedure": procedure,
                "claim_amount": claim_data['claim_amount']
            }
        }


class FinancialAnalysisAgent(BaseAgent):
    """Agent specialized in financial analysis and reserve calculation."""
    
    def __init__(self, mcp_server: MCPToolServer):
        super().__init__("Financial Analysis Agent", mcp_server)
    
    def analyze(self, claim_data: Dict) -> Dict[str, Any]:
        """
        Analyze financial aspects and recommend reserves.
        """
        claim_amount = claim_data['claim_amount']
        policy_limit = claim_data.get('policy_limit', 0)
        
        # Calculate financial metrics
        claim_to_limit_ratio = claim_amount / policy_limit if policy_limit > 0 else 0
        
        # Get claimant payment history
        payment_history = self.mcp.payment_history(claim_data['claimant_id'])
        claimant_stats = payment_history.get('summary', {})
        
        total_claimed = claimant_stats.get('total_claimed', 0)
        avg_claim = claimant_stats.get('average_claim', 0)
        
        # Financial risk assessment
        if claim_to_limit_ratio > 0.9:
            recommendation = "INVESTIGATE"
            confidence = 0.8
            rationale = f"Claim represents {claim_to_limit_ratio*100:.1f}% of policy limit - high financial exposure"
        elif claim_amount > avg_claim * 3 and avg_claim > 0:
            recommendation = "REVIEW"
            confidence = 0.7
            rationale = f"Claim ${claim_amount:,.2f} is 3x claimant's average ${avg_claim:,.2f}"
        elif total_claimed > policy_limit * 2:
            recommendation = "INVESTIGATE"
            confidence = 0.75
            rationale = f"Claimant's lifetime claims ${total_claimed:,.2f} exceed 2x policy limit"
        else:
            recommendation = "APPROVE"
            confidence = 0.85
            rationale = "Financial metrics within acceptable parameters."
        
        # Calculate recommended reserve
        reserve_amount = claim_amount * 1.1  # 10% buffer
        
        return {
            "agent": self.name,
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "analysis": {
                "claim_amount": claim_amount,
                "policy_limit": policy_limit,
                "claim_to_limit_ratio": round(claim_to_limit_ratio, 4),
                "claimant_total_claimed": total_claimed,
                "claimant_avg_claim": avg_claim,
                "recommended_reserve": round(reserve_amount, 2)
            }
        }


class ClaimsOrchestrator:
    """Central orchestrator that routes claims to specialist agents."""
    
    def __init__(self, catalog="main", schema="insurance_claims"):
        self.catalog = catalog
        self.schema = schema
        self.mcp = MCPToolServer(catalog, schema)
        
        # Initialize specialist agents
        self.agents = {
            'fraud': FraudDetectionAgent(self.mcp),
            'legal': LegalComplianceAgent(self.mcp),
            'medical': MedicalReviewAgent(self.mcp),
            'financial': FinancialAnalysisAgent(self.mcp)
        }
        
        self.spark = SparkSession.builder.getOrCreate()
        
    def route_claim(self, claim_id: str, priority: str = "normal") -> Dict[str, Any]:
        """
        Route claim to appropriate specialist agents based on characteristics.
        
        Args:
            claim_id: Unique claim identifier
            priority: "low", "normal", "high", or "critical"
        
        Returns:
            Aggregated analysis and unified recommendation
        """
        # Get claim details
        claim_details = self.mcp.get_claim_details(claim_id)
        
        if claim_details.get('status') == 'error':
            return {
                'status': 'error',
                'message': f"Claim {claim_id} not found"
            }
        
        # Extract claim data
        claim_data = {
            'claim_id': claim_id,
            'claimant_id': claim_details['claimant']['id'],
            'policy_id': claim_details['policy']['id'],
            'claim_type': claim_details['claim']['type'],
            'claim_amount': claim_details['claim']['amount'],
            'incident_date': claim_details['claim']['incident_date'],
            'filing_date': claim_details['claim']['filing_date'],
            'description': claim_details['claim']['description'],
            'diagnosis_code': claim_details['medical']['diagnosis_code'],
            'procedure_code': claim_details['medical']['procedure_code'],
            'policy_limit': claim_details['policy']['limit'],
            'status': claim_details['claim']['status']
        }
        
        # Determine which agents to consult
        agents_to_consult = ['fraud', 'legal', 'financial']
        
        # Add medical agent for health-related claims
        if claim_data['claim_type'] in ['Health', 'Workers Comp']:
            agents_to_consult.append('medical')
        
        # High priority claims get all agents
        if priority in ['high', 'critical']:
            agents_to_consult = list(self.agents.keys())
        
        # Consult each agent
        agent_responses = []
        for agent_key in agents_to_consult:
            agent = self.agents[agent_key]
            response = agent.analyze(claim_data)
            agent_responses.append(response)
        
        # Aggregate recommendations
        final_recommendation = self._aggregate_recommendations(agent_responses, priority)
        
        # Add metadata
        investigation_result = {
            'claim_id': claim_id,
            'investigation_timestamp': datetime.now().isoformat(),
            'priority': priority,
            'claim_data': claim_data,
            'agent_responses': agent_responses,
            'final_recommendation': final_recommendation['recommendation'],
            'confidence': final_recommendation['confidence'],
            'rationale': final_recommendation['rationale'],
            'next_actions': final_recommendation['next_actions']
        }
        
        # Log to Lakehouse for audit and evaluation
        self._log_investigation(investigation_result)
        
        return investigation_result
    
    def _aggregate_recommendations(self, agent_responses: List[Dict], priority: str) -> Dict[str, Any]:
        """
        Aggregate specialist recommendations into unified decision.
        """
        # Count recommendations
        recommendations = [r['recommendation'] for r in agent_responses if r['recommendation'] != 'N/A']
        
        deny_count = recommendations.count('DENY')
        investigate_count = recommendations.count('INVESTIGATE')
        review_count = recommendations.count('REVIEW')
        approve_count = recommendations.count('APPROVE')
        
        # Calculate weighted confidence
        confidences = [r['confidence'] for r in agent_responses if r['recommendation'] != 'N/A']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Decision logic
        if deny_count > 0:
            # Any DENY recommendation triggers denial
            final_rec = "DENY"
            rationale = "Denial recommended by " + ", ".join(
                [r['agent'] for r in agent_responses if r['recommendation'] == 'DENY']
            )
            next_actions = ["Send denial letter", "Document denial reason", "Close claim"]
        elif investigate_count >= 2 or (investigate_count >= 1 and priority in ['high', 'critical']):
            # Multiple INVESTIGATE or high priority
            final_rec = "INVESTIGATE"
            rationale = "Multiple agents flagged concerns requiring investigation"
            next_actions = [
                "Assign to senior adjuster",
                "Request additional documentation",
                "Conduct field investigation if needed"
            ]
        elif review_count + investigate_count > approve_count:
            # More concerns than approvals
            final_rec = "REVIEW"
            rationale = "Mixed signals from agents - manual review recommended"
            next_actions = [
                "Senior adjuster review",
                "Verify documentation",
                "Contact claimant for clarification"
            ]
        else:
            # Majority approve
            final_rec = "APPROVE"
            rationale = "Majority of agents recommend approval with no major concerns"
            next_actions = [
                "Process payment",
                "Update claim status",
                "Notify claimant of approval"
            ]
        
        return {
            'recommendation': final_rec,
            'confidence': round(avg_confidence, 3),
            'rationale': rationale,
            'next_actions': next_actions,
            'vote_summary': {
                'DENY': deny_count,
                'INVESTIGATE': investigate_count,
                'REVIEW': review_count,
                'APPROVE': approve_count
            }
        }
    
    def _log_investigation(self, investigation_result: Dict):
        """
        Log investigation to Lakehouse for audit trail and evaluation.
        """
        try:
            # Create log entry
            log_entry = {
                'claim_id': investigation_result['claim_id'],
                'investigation_timestamp': investigation_result['investigation_timestamp'],
                'priority': investigation_result['priority'],
                'final_recommendation': investigation_result['final_recommendation'],
                'confidence': investigation_result['confidence'],
                'rationale': investigation_result['rationale'],
                'agent_count': len(investigation_result['agent_responses']),
                'agents_consulted': ', '.join([r['agent'] for r in investigation_result['agent_responses']]),
                'investigation_json': json.dumps(investigation_result)
            }
            
            # Convert to Spark DataFrame
            log_df = self.spark.createDataFrame([log_entry])
            
            # Append to audit log table
            log_df.write \
                .format("delta") \
                .mode("append") \
                .saveAsTable(f"{self.catalog}.{self.schema}.agent_investigation_log")
            
        except Exception as e:
            print(f"Warning: Failed to log investigation: {e}")


def main():
    """Main entry point for multi-agent orchestrator."""
    print("Insurance Claims Multi-Agent Orchestration System")
    print("=" * 60)
    
    orchestrator = ClaimsOrchestrator()
    
    print("\nSpecialist Agents Initialized:")
    for agent_name, agent in orchestrator.agents.items():
        print(f"  • {agent.name}")
    
    print("\nOrchestrator ready to route claims.")
    print("\nExample usage:")
    print("  result = orchestrator.route_claim('CLM-2024-00123', priority='high')")
    print("  print(result['final_recommendation'])")
    print("  print(result['rationale'])")


if __name__ == "__main__":
    main()