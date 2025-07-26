"""
Module 1, Section 4: Advanced Pitfalls and Best Practices - Production Frameworks
================================================================================

This file contains runnable implementations of production-grade frameworks
for prompt engineering including monitoring, security, and deployment systems.
"""

import json
import time
import random
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    accuracy: float
    latency: float
    throughput: float
    error_rate: float
    cost_per_query: float
    user_satisfaction: float
    timestamp: str

class OutputFormatValidator:
    """
    Advanced output format validation system
    """
    
    def __init__(self):
        self.format_templates = {
            "json": {
                "example": '{"key": "value", "array": [1, 2, 3]}',
                "validation_schema": "json_schema.json",
                "error_patterns": ["trailing_comma", "unquoted_keys", "invalid_escape"]
            },
            "xml": {
                "example": '<root><item id="1">value</item></root>',
                "validation_schema": "xml_schema.xsd", 
                "error_patterns": ["unclosed_tags", "invalid_characters", "namespace_issues"]
            }
        }
    
    def generate_format_prompt(self, desired_format: str, data_description: str) -> str:
        """Generate format-specific prompt with validation"""
        template = self.format_templates.get(desired_format)
        if not template:
            raise ValueError(f"Unsupported format: {desired_format}")
        
        return f"""
        # OUTPUT FORMAT SPECIFICATION
        Return data as valid {desired_format.upper()} following this exact structure:
        
        EXAMPLE:
        {template['example']}
        
        # VALIDATION REQUIREMENTS
        - Must pass {template['validation_schema']} validation
        - Avoid common errors: {', '.join(template['error_patterns'])}
        - Include proper encoding and escaping
        
        # ERROR HANDLING
        If data cannot be formatted as requested, return:
        {{"error": "format_conversion_failed", "reason": "specific_issue"}}
        
        Data to format: {data_description}
        """

class PromptMonitoringSystem:
    """
    Production monitoring system for prompt performance
    """
    
    def __init__(self):
        self.monitoring_metrics = {
            "performance": ["accuracy", "latency", "throughput", "error_rate"],
            "cost": ["token_usage", "api_calls", "cost_per_query"],
            "quality": ["user_satisfaction", "output_quality", "consistency"],
            "security": ["safety_violations", "bias_incidents", "data_leaks"]
        }
        self.alert_thresholds = {
            "accuracy": {"min": 0.8, "severity": AlertSeverity.HIGH},
            "latency": {"max": 5.0, "severity": AlertSeverity.MEDIUM},
            "error_rate": {"max": 0.05, "severity": AlertSeverity.HIGH},
            "cost_per_query": {"max": 0.10, "severity": AlertSeverity.MEDIUM}
        }
        self.metrics_history: List[PerformanceMetrics] = []
    
    def setup_monitoring(self, prompt_id: str, baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Setup monitoring for a prompt deployment"""
        return {
            "prompt_id": prompt_id,
            "baseline": baseline_metrics,
            "alerts": self._configure_alerts(prompt_id, baseline_metrics),
            "logging": self._setup_logging(prompt_id),
            "reporting": self._configure_reports(prompt_id)
        }
    
    def real_time_analysis(self, prompt_execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform real-time analysis of prompt execution"""
        analysis = {}
        
        for metric_category, metrics in self.monitoring_metrics.items():
            category_analysis = {}
            
            for metric in metrics:
                value = prompt_execution_data.get(metric, 0.0)
                category_analysis[metric] = {
                    "current_value": value,
                    "status": self._check_threshold(metric, value),
                    "trend": self._calculate_trend(metric),
                    "alert_triggered": self._should_alert(metric, value)
                }
            
            analysis[metric_category] = category_analysis
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": self._calculate_health_score(analysis),
            "metric_breakdown": analysis,
            "recommendations": self._generate_monitoring_recommendations(analysis),
            "action_items": self._prioritize_action_items(analysis)
        }
    
    def log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _configure_alerts(self, prompt_id: str, baseline: Dict[str, float]) -> Dict[str, Any]:
        """Configure alerting system"""
        return {
            "email_alerts": True,
            "slack_integration": True,
            "thresholds": self.alert_thresholds,
            "escalation_policy": "immediate_for_critical"
        }
    
    def _setup_logging(self, prompt_id: str) -> Dict[str, str]:
        """Setup logging configuration"""
        return {
            "log_level": "INFO",
            "log_format": "structured_json",
            "retention_days": 30
        }
    
    def _configure_reports(self, prompt_id: str) -> Dict[str, Any]:
        """Configure reporting system"""
        return {
            "daily_summary": True,
            "weekly_analysis": True,
            "monthly_trends": True,
            "stakeholder_reports": ["engineering", "product", "business"]
        }
    
    def _check_threshold(self, metric: str, value: float) -> str:
        """Check if metric value exceeds thresholds"""
        threshold = self.alert_thresholds.get(metric)
        if not threshold:
            return "normal"
        
        if "min" in threshold and value < threshold["min"]:
            return "below_threshold"
        elif "max" in threshold and value > threshold["max"]:
            return "above_threshold"
        else:
            return "normal"
    
    def _calculate_trend(self, metric: str) -> str:
        """Calculate trend for metric"""
        if len(self.metrics_history) < 5:
            return "insufficient_data"
        
        recent_values = [getattr(m, metric, 0) for m in self.metrics_history[-5:]]
        if len(set(recent_values)) == 1:
            return "stable"
        
        trend = sum(recent_values[-3:]) / 3 - sum(recent_values[:2]) / 2
        
        if trend > 0.05:
            return "increasing"
        elif trend < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _should_alert(self, metric: str, value: float) -> bool:
        """Determine if alert should be triggered"""
        return self._check_threshold(metric, value) != "normal"
    
    def _calculate_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        total_metrics = 0
        healthy_metrics = 0
        
        for category, metrics in analysis.items():
            for metric, data in metrics.items():
                total_metrics += 1
                if data["status"] == "normal":
                    healthy_metrics += 1
        
        return healthy_metrics / total_metrics if total_metrics > 0 else 1.0
    
    def _generate_monitoring_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = []
        
        for category, metrics in analysis.items():
            for metric, data in metrics.items():
                if data["alert_triggered"]:
                    recommendations.append(f"Address {metric} issue in {category} category")
        
        return recommendations
    
    def _prioritize_action_items(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize action items by severity"""
        action_items = []
        
        for category, metrics in analysis.items():
            for metric, data in metrics.items():
                if data["alert_triggered"]:
                    threshold = self.alert_thresholds.get(metric, {})
                    severity = threshold.get("severity", AlertSeverity.LOW)
                    
                    action_items.append({
                        "metric": metric,
                        "category": category,
                        "severity": severity.value,
                        "current_value": data["current_value"],
                        "trend": data["trend"]
                    })
        
        # Sort by severity
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        action_items.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
        
        return action_items

class DeploymentFramework:
    """
    Production deployment framework with rollback capabilities
    """
    
    def __init__(self):
        self.rollout_phases = {
            "canary": {"traffic_percentage": 5, "duration_hours": 2},
            "blue_green": {"traffic_percentage": 50, "duration_hours": 4},
            "full_rollout": {"traffic_percentage": 100, "duration_hours": 24}
        }
    
    def execute_rollout(self, prompt_version: str, monitoring_system: PromptMonitoringSystem) -> Dict[str, Any]:
        """Execute phased rollout with monitoring"""
        rollout_results = {}
        
        for phase_name, phase_config in self.rollout_phases.items():
            phase_result = self._execute_phase(prompt_version, phase_config, monitoring_system)
            rollout_results[phase_name] = phase_result
            
            if not phase_result["success"]:
                return {
                    "rollout_status": "failed",
                    "failed_phase": phase_name,
                    "failure_reason": phase_result["failure_reason"],
                    "rollback_initiated": True,
                    "results": rollout_results
                }
        
        return {
            "rollout_status": "completed",
            "all_phases_successful": True,
            "results": rollout_results
        }
    
    def _execute_phase(self, version: str, config: Dict[str, Any], monitor: PromptMonitoringSystem) -> Dict[str, Any]:
        """Execute a single rollout phase"""
        # Simulate phase execution
        print(f"Executing phase with {config['traffic_percentage']}% traffic for {config['duration_hours']} hours")
        
        # Simulate monitoring during phase
        simulated_metrics = PerformanceMetrics(
            accuracy=0.85 + random.uniform(-0.1, 0.1),
            latency=1.5 + random.uniform(-0.5, 0.5),
            throughput=100 + random.uniform(-20, 20),
            error_rate=0.02 + random.uniform(-0.01, 0.02),
            cost_per_query=0.05 + random.uniform(-0.01, 0.02),
            user_satisfaction=4.2 + random.uniform(-0.5, 0.5),
            timestamp=datetime.now().isoformat()
        )
        
        monitor.log_metrics(simulated_metrics)
        
        # Check if phase should continue
        health_score = random.uniform(0.7, 1.0)  # Simulate health check
        
        if health_score < 0.8:
            return {
                "success": False,
                "failure_reason": f"Health score {health_score:.2f} below threshold",
                "metrics": simulated_metrics
            }
        
        return {
            "success": True,
            "health_score": health_score,
            "metrics": simulated_metrics
        }

def demonstration_examples():
    """
    Demonstrate production frameworks
    """
    print("=== Production Frameworks Demonstration ===")
    print("=" * 60)
    
    # 1. Output Format Validation
    print("1. Output Format Validation")
    print("-" * 40)
    
    validator = OutputFormatValidator()
    
    json_prompt = validator.generate_format_prompt("json", "customer support ticket data")
    print("JSON Format Prompt Generated:")
    print(json_prompt[:200] + "...")
    
    print("\n" + "=" * 60)
    
    # 2. Monitoring System
    print("2. Production Monitoring System")
    print("-" * 40)
    
    monitor = PromptMonitoringSystem()
    
    # Setup monitoring
    baseline = {"accuracy": 0.85, "latency": 2.0, "error_rate": 0.02}
    monitoring_config = monitor.setup_monitoring("prompt_v1", baseline)
    print(f"Monitoring configured for prompt_v1")
    print(f"Alert thresholds: {len(monitor.alert_thresholds)} metrics")
    
    # Simulate real-time analysis
    execution_data = {
        "accuracy": 0.75,  # Below threshold
        "latency": 3.0,
        "error_rate": 0.08,  # Above threshold
        "user_satisfaction": 4.0
    }
    
    analysis = monitor.real_time_analysis(execution_data)
    print(f"\nReal-time Analysis:")
    print(f"Overall Health Score: {analysis['overall_health']:.3f}")
    print(f"Action Items: {len(analysis['action_items'])}")
    
    for item in analysis['action_items'][:2]:  # Show top 2
        print(f"  - {item['metric']}: {item['severity']} severity")
    
    print("\n" + "=" * 60)
    
    # 3. Deployment Framework
    print("3. Production Deployment Framework")
    print("-" * 40)
    
    deployment = DeploymentFramework()
    
    print("Executing phased rollout...")
    rollout_result = deployment.execute_rollout("prompt_v2", monitor)
    
    print(f"Rollout Status: {rollout_result['rollout_status']}")
    print(f"Phases Completed: {len(rollout_result['results'])}")
    
    for phase, result in rollout_result['results'].items():
        status = "✓" if result['success'] else "✗"
        print(f"  {phase}: {status}")
    
    print("\n" + "=" * 60)
    print("Production frameworks demonstration completed!")

def main():
    """
    Main function to run production framework examples
    """
    print("Testing Module 1, Section 4: Advanced Pitfalls and Best Practices")
    print("=" * 80)
    
    # Run demonstration
    demonstration_examples()
    
    print("\n" + "=" * 80)
    print("All production framework examples completed successfully!")

if __name__ == "__main__":
    main()
