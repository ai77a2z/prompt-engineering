"""
Module 1, Section 3: Professional Workflow - Workflow Management
===============================================================

This file contains runnable implementations of professional development
practices including version control, team collaboration, and agile
prompt engineering workflows.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"

@dataclass
class PromptVersion:
    """Data class for prompt versions"""
    id: str
    timestamp: str
    content: str
    author: str
    description: str
    performance_metrics: Dict[str, float]
    test_results: Dict[str, Any]
    deployment_status: DeploymentStatus
    parent_version: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class PromptVersionManager:
    """
    Advanced version control system for prompts
    """
    
    def __init__(self, repository_path: str = "./prompt_repo"):
        self.repo_path = repository_path
        self.version_history: List[PromptVersion] = []
        self.branches: Dict[str, List[str]] = {"main": []}
        self.current_branch = "main"
    
    def create_version(self, prompt_content: str, metadata: Dict[str, Any]) -> str:
        """
        Create a new prompt version with comprehensive metadata
        
        Args:
            prompt_content: The prompt text
            metadata: Version metadata including author, description, metrics
            
        Returns:
            Version ID
        """
        version_id = self._generate_version_id()
        
        version = PromptVersion(
            id=version_id,
            timestamp=datetime.now().isoformat(),
            content=prompt_content,
            author=metadata.get("author", "unknown"),
            description=metadata.get("description", ""),
            performance_metrics=metadata.get("metrics", {}),
            test_results=metadata.get("test_results", {}),
            deployment_status=DeploymentStatus(metadata.get("deployment_status", "development")),
            parent_version=metadata.get("parent_version"),
            tags=metadata.get("tags", [])
        )
        
        self.version_history.append(version)
        self.branches[self.current_branch].append(version_id)
        self._save_to_repository(version)
        
        return version_id
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        Compare two prompt versions comprehensively
        
        Args:
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Detailed comparison results
        """
        v1 = self._get_version(version1_id)
        v2 = self._get_version(version2_id)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        return {
            "content_diff": self._calculate_content_diff(v1.content, v2.content),
            "performance_comparison": self._compare_metrics(v1.performance_metrics, v2.performance_metrics),
            "deployment_status_change": {
                "from": v1.deployment_status.value,
                "to": v2.deployment_status.value
            },
            "time_difference": self._calculate_time_diff(v1.timestamp, v2.timestamp),
            "recommendation": self._recommend_version(v1, v2)
        }
    
    def create_branch(self, branch_name: str, from_version: Optional[str] = None) -> bool:
        """Create a new development branch"""
        if branch_name in self.branches:
            return False
        
        if from_version:
            # Create branch from specific version
            self.branches[branch_name] = [from_version]
        else:
            # Create branch from current HEAD
            current_versions = self.branches[self.current_branch]
            self.branches[branch_name] = current_versions.copy() if current_versions else []
        
        return True
    
    def merge_branch(self, source_branch: str, target_branch: str = "main") -> Dict[str, Any]:
        """Merge one branch into another"""
        if source_branch not in self.branches or target_branch not in self.branches:
            return {"success": False, "error": "Branch not found"}
        
        source_versions = self.branches[source_branch]
        target_versions = self.branches[target_branch]
        
        # Simple merge strategy: add new versions from source
        new_versions = [v for v in source_versions if v not in target_versions]
        self.branches[target_branch].extend(new_versions)
        
        return {
            "success": True,
            "merged_versions": len(new_versions),
            "target_branch": target_branch,
            "source_branch": source_branch
        }
    
    def get_version_history(self, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get version history for a branch"""
        branch = branch or self.current_branch
        
        if branch not in self.branches:
            return []
        
        version_ids = self.branches[branch]
        versions = [self._get_version(vid) for vid in version_ids if self._get_version(vid)]
        
        return [
            {
                "id": v.id,
                "timestamp": v.timestamp,
                "author": v.author,
                "description": v.description,
                "deployment_status": v.deployment_status.value,
                "performance_summary": self._summarize_metrics(v.performance_metrics),
                "tags": v.tags
            }
            for v in versions
        ]
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        return f"v_{uuid.uuid4().hex[:8]}"
    
    def _get_version(self, version_id: str) -> Optional[PromptVersion]:
        """Get version by ID"""
        for version in self.version_history:
            if version.id == version_id:
                return version
        return None
    
    def _calculate_content_diff(self, content1: str, content2: str) -> Dict[str, Any]:
        """Calculate content differences"""
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')
        
        # Simple diff calculation
        added_lines = len(lines2) - len(lines1)
        
        # Calculate similarity
        similarity = self._calculate_similarity(content1, content2)
        
        return {
            "similarity_score": similarity,
            "lines_added": max(0, added_lines),
            "lines_removed": max(0, -added_lines),
            "character_difference": len(content2) - len(content1),
            "significant_change": similarity < 0.8
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _compare_metrics(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> Dict[str, Any]:
        """Compare performance metrics"""
        comparison = {}
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric, 0.0)
            val2 = metrics2.get(metric, 0.0)
            
            comparison[metric] = {
                "before": val1,
                "after": val2,
                "change": val2 - val1,
                "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0.0,
                "improvement": val2 > val1
            }
        
        return comparison
    
    def _calculate_time_diff(self, timestamp1: str, timestamp2: str) -> Dict[str, Any]:
        """Calculate time difference between versions"""
        try:
            dt1 = datetime.fromisoformat(timestamp1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(timestamp2.replace('Z', '+00:00'))
            
            diff = dt2 - dt1
            
            return {
                "days": diff.days,
                "hours": diff.seconds // 3600,
                "minutes": (diff.seconds % 3600) // 60,
                "total_seconds": diff.total_seconds()
            }
        except:
            return {"error": "Invalid timestamp format"}
    
    def _recommend_version(self, v1: PromptVersion, v2: PromptVersion) -> str:
        """Recommend which version to use"""
        # Simple recommendation logic
        v1_score = sum(v1.performance_metrics.values()) / len(v1.performance_metrics) if v1.performance_metrics else 0.5
        v2_score = sum(v2.performance_metrics.values()) / len(v2.performance_metrics) if v2.performance_metrics else 0.5
        
        if v2_score > v1_score + 0.1:
            return f"Recommend version {v2.id} - significant performance improvement"
        elif v1_score > v2_score + 0.1:
            return f"Recommend version {v1.id} - better performance"
        else:
            return f"Versions have similar performance - consider other factors"
    
    def _summarize_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Summarize performance metrics"""
        if not metrics:
            return {"average": 0.0, "count": 0}
        
        return {
            "average": sum(metrics.values()) / len(metrics),
            "count": len(metrics),
            "best_metric": max(metrics.items(), key=lambda x: x[1]) if metrics else None,
            "worst_metric": min(metrics.items(), key=lambda x: x[1]) if metrics else None
        }
    
    def _save_to_repository(self, version: PromptVersion):
        """Save version to repository (simulated)"""
        # In production, this would save to actual storage
        print(f"Saved version {version.id} to repository")

class PromptDevelopmentSprint:
    """
    Agile sprint management for prompt engineering
    """
    
    def __init__(self, sprint_duration: int = 2):
        self.sprint_duration = sprint_duration  # weeks
        self.team_roles = [
            "prompt_engineer", "product_manager", "software_engineer",
            "data_scientist", "ux_researcher", "security_engineer", "domain_expert"
        ]
        self.deliverables = {
            "week_1": {
                "prompt_engineer": ["Initial prompt versions", "Basic evaluation framework"],
                "data_scientist": ["Evaluation metrics design", "Statistical test plan"],
                "product_manager": ["Success criteria definition", "Test case requirements"],
                "domain_expert": ["Domain validation criteria", "Expert test cases"]
            },
            "week_2": {
                "prompt_engineer": ["Refined prompt versions", "Performance analysis"],
                "software_engineer": ["Integration prototype", "Monitoring setup"],
                "ux_researcher": ["User feedback analysis", "Usability recommendations"],
                "security_engineer": ["Security assessment", "Safety validation"]
            }
        }
    
    def generate_sprint_plan(self, project_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive sprint plan"""
        return {
            "sprint_goal": project_requirements["objective"],
            "duration_weeks": self.sprint_duration,
            "team_composition": self.team_roles,
            "weekly_deliverables": self.deliverables,
            "success_metrics": project_requirements.get("success_metrics", []),
            "risk_mitigation": self._identify_risks(project_requirements),
            "review_schedule": self._create_review_schedule()
        }
    
    def track_sprint_progress(self, completed_deliverables: Dict[str, List[str]]) -> Dict[str, Any]:
        """Track sprint progress against plan"""
        total_deliverables = 0
        completed_count = 0
        
        for week, roles in self.deliverables.items():
            for role, deliverables in roles.items():
                total_deliverables += len(deliverables)
                completed_count += len(completed_deliverables.get(f"{week}_{role}", []))
        
        progress_percentage = (completed_count / total_deliverables) * 100 if total_deliverables > 0 else 0
        
        return {
            "progress_percentage": progress_percentage,
            "completed_deliverables": completed_count,
            "total_deliverables": total_deliverables,
            "on_track": progress_percentage >= 50,  # Simplified threshold
            "recommendations": self._generate_progress_recommendations(progress_percentage)
        }
    
    def _identify_risks(self, requirements: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify potential project risks"""
        risks = [
            {
                "risk": "Scope creep",
                "mitigation": "Clear requirements documentation and change control process",
                "probability": "medium"
            },
            {
                "risk": "Technical complexity underestimation",
                "mitigation": "Technical spike and proof-of-concept development",
                "probability": "high"
            },
            {
                "risk": "Team coordination challenges",
                "mitigation": "Daily standups and clear communication channels",
                "probability": "medium"
            }
        ]
        
        # Add domain-specific risks based on requirements
        if requirements.get("domain") == "healthcare":
            risks.append({
                "risk": "Regulatory compliance complexity",
                "mitigation": "Early compliance review and expert consultation",
                "probability": "high"
            })
        
        return risks
    
    def _create_review_schedule(self) -> List[Dict[str, str]]:
        """Create review and milestone schedule"""
        return [
            {
                "event": "Sprint Planning",
                "timing": "Week 0 - Monday",
                "participants": "All team members",
                "duration": "2 hours"
            },
            {
                "event": "Mid-Sprint Review",
                "timing": "Week 1 - Friday",
                "participants": "Core team + stakeholders",
                "duration": "1 hour"
            },
            {
                "event": "Sprint Demo",
                "timing": "Week 2 - Thursday",
                "participants": "All team members + stakeholders",
                "duration": "1.5 hours"
            },
            {
                "event": "Sprint Retrospective",
                "timing": "Week 2 - Friday",
                "participants": "Core team only",
                "duration": "1 hour"
            }
        ]
    
    def _generate_progress_recommendations(self, progress: float) -> List[str]:
        """Generate recommendations based on progress"""
        if progress < 25:
            return [
                "Consider reducing scope or extending timeline",
                "Identify and address blocking issues immediately",
                "Increase team collaboration and communication"
            ]
        elif progress < 50:
            return [
                "Focus on critical path deliverables",
                "Consider parallel work streams where possible",
                "Schedule additional check-ins with stakeholders"
            ]
        elif progress < 75:
            return [
                "Maintain current pace",
                "Prepare for final sprint push",
                "Begin planning next sprint activities"
            ]
        else:
            return [
                "Excellent progress - maintain momentum",
                "Begin documentation and knowledge transfer",
                "Plan celebration and team recognition"
            ]

def demonstration_examples():
    """
    Demonstrate workflow management capabilities
    """
    print("=== Professional Workflow Management Demonstration ===")
    print("=" * 60)
    
    # 1. Version Control System
    print("1. Version Control System")
    print("-" * 40)
    
    version_manager = PromptVersionManager()
    
    # Create initial version
    initial_prompt = """
    Extract product information from text.
    Return as JSON.
    """
    
    v1_id = version_manager.create_version(
        initial_prompt,
        {
            "author": "alice@company.com",
            "description": "Initial basic prompt",
            "metrics": {"accuracy": 0.65, "efficiency": 0.8},
            "test_results": {"passed": 5, "failed": 3}
        }
    )
    
    # Create improved version
    improved_prompt = """
    # ROLE & EXPERTISE
    You are an expert data extraction specialist.
    
    # TASK
    Extract product information from the provided text with high accuracy.
    
    # OUTPUT FORMAT
    Return a valid JSON object with product details.
    
    # CONSTRAINTS
    - Only extract explicitly mentioned products
    - Validate all data before output
    """
    
    v2_id = version_manager.create_version(
        improved_prompt,
        {
            "author": "bob@company.com",
            "description": "Enhanced prompt with role and constraints",
            "metrics": {"accuracy": 0.85, "efficiency": 0.75},
            "test_results": {"passed": 8, "failed": 1},
            "parent_version": v1_id,
            "tags": ["enhanced", "production-candidate"]
        }
    )
    
    print(f"Created versions: {v1_id}, {v2_id}")
    
    # Compare versions
    comparison = version_manager.compare_versions(v1_id, v2_id)
    print(f"\nVersion Comparison:")
    print(f"Content similarity: {comparison['content_diff']['similarity_score']:.3f}")
    print(f"Performance changes:")
    for metric, change in comparison['performance_comparison'].items():
        print(f"  {metric}: {change['before']:.3f} → {change['after']:.3f} ({change['change']:+.3f})")
    print(f"Recommendation: {comparison['recommendation']}")
    
    # Version history
    history = version_manager.get_version_history()
    print(f"\nVersion History ({len(history)} versions):")
    for version in history:
        print(f"  {version['id']}: {version['description']} by {version['author']}")
    
    print("\n" + "=" * 60)
    
    # 2. Sprint Management
    print("2. Agile Sprint Management")
    print("-" * 40)
    
    sprint_manager = PromptDevelopmentSprint()
    
    # Create sprint plan
    project_requirements = {
        "objective": "Develop customer support ticket classification system",
        "domain": "customer_service",
        "success_metrics": ["accuracy > 0.9", "response_time < 2s", "user_satisfaction > 4.5"]
    }
    
    sprint_plan = sprint_manager.generate_sprint_plan(project_requirements)
    
    print(f"Sprint Goal: {sprint_plan['sprint_goal']}")
    print(f"Duration: {sprint_plan['duration_weeks']} weeks")
    print(f"Team Size: {len(sprint_plan['team_composition'])} roles")
    
    print("\nWeek 1 Deliverables:")
    for role, deliverables in sprint_plan['weekly_deliverables']['week_1'].items():
        print(f"  {role}: {', '.join(deliverables)}")
    
    print(f"\nIdentified Risks: {len(sprint_plan['risk_mitigation'])}")
    for risk in sprint_plan['risk_mitigation'][:2]:  # Show first 2
        print(f"  - {risk['risk']} (Probability: {risk['probability']})")
        print(f"    Mitigation: {risk['mitigation']}")
    
    # Track progress
    completed_work = {
        "week_1_prompt_engineer": ["Initial prompt versions"],
        "week_1_data_scientist": ["Evaluation metrics design"],
        "week_1_product_manager": ["Success criteria definition"]
    }
    
    progress = sprint_manager.track_sprint_progress(completed_work)
    print(f"\nSprint Progress: {progress['progress_percentage']:.1f}%")
    print(f"Status: {'On Track' if progress['on_track'] else 'Behind Schedule'}")
    print("Recommendations:")
    for rec in progress['recommendations'][:2]:
        print(f"  - {rec}")
    
    print("\n" + "=" * 60)
    print("Workflow management demonstration completed!")

def main():
    """
    Main function to run workflow management examples
    """
    print("Testing Module 1, Section 3: Professional Workflow - Workflow Management")
    print("=" * 80)
    
    # Run demonstration
    demonstration_examples()
    
    print("\n" + "=" * 80)
    print("All workflow management examples completed successfully!")

if __name__ == "__main__":
    main()
