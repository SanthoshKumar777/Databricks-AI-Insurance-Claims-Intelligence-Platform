#!/usr/bin/env python3
"""
Evaluation Framework for Multi-Agent Claims Intelligence System

Provides:
1. Human Feedback Collection - Adjuster ratings and corrections
2. A/B Testing - Compare routing strategies and agent performance
3. Quality Metrics - Track precision, recall, F1 for recommendations
4. Performance Dashboard - Real-time monitoring of agent accuracy
5. Continuous Learning - Feed back evaluation data to improve models

All evaluation data stored in Lakehouse for analysis and model improvement.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import mlflow


class FeedbackCollector:
    """Collects and stores human feedback on agent recommendations."""
    
    def __init__(self, catalog="main", schema="insurance_claims"):
        self.catalog = catalog
        self.schema = schema
        self.spark = SparkSession.builder.getOrCreate()
        self.feedback_table = f"{catalog}.{schema}.agent_feedback"
        
        # Ensure feedback table exists
        self._initialize_feedback_table()
    
    def _initialize_feedback_table(self):
        """Create feedback table if it doesn't exist."""
        schema = StructType([
            StructField('feedback_id', StringType(), False),
            StructField('claim_id', StringType(), False),
            StructField('agent_recommendation', StringType(), True),
            StructField('adjuster_decision', StringType(), True),
            StructField('adjuster_notes', StringType(), True),
            StructField('rating', IntegerType(), True),
            StructField('feedback_timestamp', TimestampType(), True),
            StructField('adjuster_id', StringType(), True),
            StructField('time_to_decision_hours', DoubleType(), True),
            StructField('recommendation_correct', BooleanType(), True)
        ])
        
        # Create table if not exists
        try:
            self.spark.sql(f"""
                CREATE TABLE IF NOT EXISTS {self.feedback_table} (
                    feedback_id STRING,
                    claim_id STRING,
                    agent_recommendation STRING,
                    adjuster_decision STRING,
                    adjuster_notes STRING,
                    rating INT,
                    feedback_timestamp TIMESTAMP,
                    adjuster_id STRING,
                    time_to_decision_hours DOUBLE,
                    recommendation_correct BOOLEAN
                )
                USING DELTA
            """)
        except Exception as e:
            print(f"Feedback table already exists or error: {e}")
    
    def submit_feedback(
        self,
        claim_id: str,
        agent_recommendation: str,
        adjuster_decision: str,
        adjuster_notes: str = "",
        rating: int = 3,
        adjuster_id: Optional[str] = None,
        time_to_decision_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit adjuster feedback on agent recommendation.
        
        Args:
            claim_id: Claim identifier
            agent_recommendation: Agent's recommended action
            adjuster_decision: Adjuster's final decision
            adjuster_notes: Explanation for decision
            rating: 1-5 rating of recommendation quality
            adjuster_id: Adjuster who made decision
            time_to_decision_hours: Time taken to reach decision
        
        Returns:
            Feedback record
        """
        import uuid
        
        feedback_id = str(uuid.uuid4())
        recommendation_correct = (agent_recommendation.upper() == adjuster_decision.upper())
        
        feedback_record = {
            'feedback_id': feedback_id,
            'claim_id': claim_id,
            'agent_recommendation': agent_recommendation.upper(),
            'adjuster_decision': adjuster_decision.upper(),
            'adjuster_notes': adjuster_notes,
            'rating': max(1, min(5, rating)),  # Clamp to 1-5
            'feedback_timestamp': datetime.now(),
            'adjuster_id': adjuster_id or 'anonymous',
            'time_to_decision_hours': time_to_decision_hours,
            'recommendation_correct': recommendation_correct
        }
        
        # Save to Lakehouse
        feedback_df = self.spark.createDataFrame([feedback_record])
        feedback_df.write.format("delta").mode("append").saveAsTable(self.feedback_table)
        
        print(f"✓ Feedback submitted for claim {claim_id}")
        print(f"  Agent recommended: {agent_recommendation}")
        print(f"  Adjuster decided: {adjuster_decision}")
        print(f"  Match: {recommendation_correct}")
        
        return feedback_record
    
    def get_feedback_summary(self, days: int = 30) -> pd.DataFrame:
        """
        Get feedback summary for recent period.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        summary_df = self.spark.sql(f"""
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                SUM(CAST(recommendation_correct AS INT)) as correct_recommendations,
                COUNT(*) - SUM(CAST(recommendation_correct AS INT)) as incorrect_recommendations,
                SUM(CAST(recommendation_correct AS INT)) / COUNT(*) * 100 as accuracy_pct,
                AVG(time_to_decision_hours) as avg_decision_time_hours,
                agent_recommendation,
                COUNT(*) as recommendation_count
            FROM {self.feedback_table}
            WHERE feedback_timestamp >= '{cutoff_date.strftime('%Y-%m-%d')}'
            GROUP BY agent_recommendation
            ORDER BY recommendation_count DESC
        """)
        
        return summary_df.toPandas()


class ABTestManager:
    """Manages A/B testing of different routing strategies."""
    
    def __init__(self, catalog="main", schema="insurance_claims"):
        self.catalog = catalog
        self.schema = schema
        self.spark = SparkSession.builder.getOrCreate()
        self.experiments_table = f"{catalog}.{schema}.ab_test_experiments"
    
    def create_experiment(
        self,
        experiment_name: str,
        variant_a: Dict[str, Any],
        variant_b: Dict[str, Any],
        allocation_ratio: float = 0.5,
        description: str = ""
    ) -> str:
        """
        Create new A/B test experiment.
        
        Args:
            experiment_name: Name of experiment
            variant_a: Configuration for variant A (control)
            variant_b: Configuration for variant B (treatment)
            allocation_ratio: Fraction allocated to B (0.5 = 50/50 split)
            description: Experiment description
        
        Returns:
            Experiment ID
        """
        import uuid
        
        experiment_id = str(uuid.uuid4())
        
        experiment_record = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'variant_a_config': json.dumps(variant_a),
            'variant_b_config': json.dumps(variant_b),
            'allocation_ratio': allocation_ratio,
            'description': description,
            'start_timestamp': datetime.now(),
            'status': 'ACTIVE'
        }
        
        # Save to experiments table
        exp_df = self.spark.createDataFrame([experiment_record])
        exp_df.write.format("delta").mode("append").saveAsTable(self.experiments_table)
        
        print(f"✓ Created A/B test experiment: {experiment_name}")
        print(f"  Experiment ID: {experiment_id}")
        print(f"  Allocation: {allocation_ratio*100:.0f}% to variant B")
        
        return experiment_id
    
    def assign_variant(self, claim_id: str, experiment_id: str) -> str:
        """
        Assign claim to variant A or B using consistent hashing.
        """
        import hashlib
        
        # Get experiment config
        exp_df = self.spark.table(self.experiments_table).filter(
            col('experiment_id') == experiment_id
        ).first()
        
        if not exp_df:
            return 'A'  # Default to control
        
        allocation_ratio = exp_df['allocation_ratio']
        
        # Consistent hash-based assignment
        hash_value = int(hashlib.md5(claim_id.encode()).hexdigest(), 16)
        normalized = (hash_value % 1000) / 1000.0
        
        return 'B' if normalized < allocation_ratio else 'A'
    
    def log_experiment_result(
        self,
        experiment_id: str,
        claim_id: str,
        variant: str,
        outcome: Dict[str, Any]
    ):
        """
        Log result for A/B test assignment.
        """
        result_record = {
            'experiment_id': experiment_id,
            'claim_id': claim_id,
            'variant': variant,
            'outcome_json': json.dumps(outcome),
            'timestamp': datetime.now()
        }
        
        # Append to results table
        result_df = self.spark.createDataFrame([result_record])
        result_df.write.format("delta").mode("append").saveAsTable(
            f"{self.catalog}.{self.schema}.ab_test_results"
        )


class QualityMetrics:
    """Calculates quality metrics for agent recommendations."""
    
    def __init__(self, catalog="main", schema="insurance_claims"):
        self.catalog = catalog
        self.schema = schema
        self.spark = SparkSession.builder.getOrCreate()
    
    def calculate_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate precision, recall, F1 for agent recommendations.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get feedback data
        feedback_df = self.spark.table(f"{self.catalog}.{self.schema}.agent_feedback").filter(
            col('feedback_timestamp') >= lit(cutoff_date)
        )
        
        # Convert to pandas for sklearn metrics
        feedback_pd = feedback_df.toPandas()
        
        if len(feedback_pd) == 0:
            return {
                'status': 'no_data',
                'message': f'No feedback data in last {days} days'
            }
        
        # Calculate confusion matrix components
        # For multi-class (APPROVE, DENY, INVESTIGATE, REVIEW)
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        
        y_true = feedback_pd['adjuster_decision']
        y_pred = feedback_pd['agent_recommendation']
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate time metrics
        avg_decision_time = feedback_pd['time_to_decision_hours'].mean()
        avg_rating = feedback_pd['rating'].mean()
        
        return {
            'period_days': days,
            'total_evaluations': len(feedback_pd),
            'accuracy': round(accuracy, 4),
            'avg_rating': round(avg_rating, 2),
            'avg_decision_time_hours': round(avg_decision_time, 2) if pd.notna(avg_decision_time) else None,
            'classification_report': report,
            'per_class_metrics': {
                label: {
                    'precision': round(metrics['precision'], 4),
                    'recall': round(metrics['recall'], 4),
                    'f1_score': round(metrics['f1-score'], 4),
                    'support': int(metrics['support'])
                }
                for label, metrics in report.items()
                if label not in ['accuracy', 'macro avg', 'weighted avg']
            }
        }
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate detailed performance report.
        """
        # Aggregate metrics by agent and recommendation type
        report_df = self.spark.sql(f"""
            SELECT 
                f.agent_recommendation,
                f.adjuster_decision,
                COUNT(*) as count,
                AVG(f.rating) as avg_rating,
                SUM(CAST(f.recommendation_correct AS INT)) / COUNT(*) * 100 as accuracy_pct,
                AVG(f.time_to_decision_hours) as avg_decision_hours
            FROM {self.catalog}.{self.schema}.agent_feedback f
            WHERE f.feedback_timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
            GROUP BY f.agent_recommendation, f.adjuster_decision
            ORDER BY count DESC
        """)
        
        report_pd = report_df.toPandas()
        
        if output_path:
            report_pd.to_csv(output_path, index=False)
            print(f"✓ Performance report saved to {output_path}")
        
        return report_pd


class PerformanceDashboard:
    """Real-time monitoring dashboard data provider."""
    
    def __init__(self, catalog="main", schema="insurance_claims"):
        self.catalog = catalog
        self.schema = schema
        self.spark = SparkSession.builder.getOrCreate()
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics for monitoring dashboard.
        """
        # Last 24 hours metrics
        metrics_24h = self.spark.sql(f"""
            SELECT 
                COUNT(*) as total_claims_investigated,
                AVG(confidence) as avg_confidence,
                final_recommendation,
                COUNT(*) as recommendation_count
            FROM {self.catalog}.{self.schema}.agent_investigation_log
            WHERE investigation_timestamp >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
            GROUP BY final_recommendation
        """).toPandas()
        
        # Feedback metrics
        feedback_24h = self.spark.sql(f"""
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                SUM(CAST(recommendation_correct AS INT)) / COUNT(*) * 100 as accuracy_pct
            FROM {self.catalog}.{self.schema}.agent_feedback
            WHERE feedback_timestamp >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
        """).toPandas()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'period': 'Last 24 hours',
            'investigations': metrics_24h.to_dict('records'),
            'feedback_summary': feedback_24h.to_dict('records')[0] if len(feedback_24h) > 0 else {},
            'status': 'healthy'
        }


def main():
    """Main entry point for evaluation framework."""
    print("Evaluation Framework for Multi-Agent Claims Intelligence")
    print("=" * 60)
    
    # Initialize components
    feedback = FeedbackCollector()
    ab_test = ABTestManager()
    metrics = QualityMetrics()
    dashboard = PerformanceDashboard()
    
    print("\n✓ Feedback collector initialized")
    print("✓ A/B test manager initialized")
    print("✓ Quality metrics calculator initialized")
    print("✓ Performance dashboard initialized")
    
    print("\nFramework ready for evaluation and continuous improvement.")
    print("\nExample usage:")
    print("  # Submit feedback")
    print("  feedback.submit_feedback('CLM-2024-00123', 'DENY', 'APPROVE', rating=4)")
    print("\n  # Calculate metrics")
    print("  metrics_report = metrics.calculate_metrics(days=30)")
    print("  print(metrics_report['accuracy'])")
    print("\n  # Get dashboard data")
    print("  dashboard_data = dashboard.get_dashboard_metrics()")


if __name__ == "__main__":
    main()