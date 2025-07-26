"""
Module 1, Section 1: Introduction to Prompting - Basic Examples
==============================================================

This file contains runnable examples from the Introduction section,
testing and validating the embedded code snippets.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Example 1: Basic Extraction Prompt
def basic_extraction_example():
    """
    Basic example of data extraction from chat logs
    """
    # Simple extraction prompt
    prompt = """
Extract all product names and their corresponding prices from the following chat log.
Return the result as a valid JSON object with keys "product_name" and "price".

Chat Log
"User: Hi, I'd like to order the A1 Power Bank. How much is it?
Agent: The A1 Power Bank is $29.99. We also have the X5 Cable for $9.95.
User: I'll just take the power bank, thanks."

JSON Output:
"""
    
    print("=== Basic Extraction Prompt ===")
    print(prompt)
    print("\nExpected Output: JSON with product names and prices")
    return prompt

# Example 2: Advanced Prompt with Cognitive Architecture
def advanced_extraction_example():
    """
    Advanced example with reasoning and error handling
    """
    advanced_prompt = """
# ROLE & EXPERTISE
You are an expert data extraction specialist with deep knowledge of e-commerce conversations and pricing patterns.

# COGNITIVE APPROACH
Use the following reasoning process:
1. Parse the conversation chronologically
2. Identify product mentions and associated price statements
3. Validate price formats and currency
4. Cross-reference to ensure accuracy
5. Handle ambiguities with explicit reasoning

# TASK
Extract all product names and their corresponding prices from the chat log below.

# CONSTRAINTS
- Only extract explicitly mentioned prices
- Ignore implied or suggested prices
- Handle currency variations (USD assumed if not specified)
- Flag any ambiguous cases

# OUTPUT FORMAT
Return a JSON object with this exact structure:
{
  "products": [
    {
      "product_name": "exact name as mentioned",
      "price": "numerical value",
      "currency": "USD",
      "confidence": "high|medium|low"
    }
  ],
  "ambiguities": ["list any unclear cases"],
  "reasoning": "brief explanation of extraction logic"
}

# ERROR HANDLING
If no products/prices found, return: {"products": [], "message": "No pricing information detected"}

# CHAT LOG
"User: Hi, I'd like to order the A1 Power Bank. How much is it?
Agent: The A1 Power Bank is $29.99. We also have the X5 Cable for $9.95.
User: I'll just take the power bank, thanks."

# BEGIN EXTRACTION
"""
    
    print("=== Advanced Extraction Prompt ===")
    print(advanced_prompt)
    print("\nExpected Output: Structured JSON with confidence scores and reasoning")
    return advanced_prompt

# Example 3: Meta-Prompting
def meta_prompting_example():
    """
    Example of prompts that generate other prompts
    """
    meta_prompt = """
You are a prompt engineering expert. Generate an optimized prompt for the following task:

Task: Classify customer support tickets by urgency (low, medium, high, critical)
Current prompt issues: Too many false positives for "critical" classification
Desired improvements: Better precision, clearer criteria, consistent formatting

Generate a production-ready prompt that addresses these issues.
"""
    
    print("=== Meta-Prompting Example ===")
    print(meta_prompt)
    print("\nExpected Output: An optimized prompt for ticket classification")
    return meta_prompt

# Example 4: Prompt Template System
class PromptTemplate:
    """
    Advanced prompt template system with validation
    """
    def __init__(self, base_template: str, validation_rules: Dict[str, Any] = None):
        self.base_template = base_template
        self.validation_rules = validation_rules or {}
    
    def format(self, **kwargs) -> str:
        # Validate inputs
        for var, rule in self.validation_rules.items():
            if var in kwargs:
                assert rule(kwargs[var]), f"Validation failed for {var}"
        
        return self.base_template.format(**kwargs)

def prompt_template_example():
    """
    Example of using the PromptTemplate class
    """
    # Usage
    analysis_template = PromptTemplate(
        base_template="""
        You are a {expertise_level} {domain} analyst.
        Analyze the following {data_type} using {methodology}.
        
        Data: {input_data}
        
        Provide analysis in {output_format} format.
        """,
        validation_rules={
            "expertise_level": lambda x: x in ["junior", "senior", "expert"],
            "domain": lambda x: len(x) > 0,
            "output_format": lambda x: x in ["JSON", "XML", "markdown", "plain_text"]
        }
    )
    
    print("=== Prompt Template Example ===")
    print("Template created successfully")
    
    # Test the template
    try:
        formatted_prompt = analysis_template.format(
            expertise_level="senior",
            domain="financial",
            data_type="quarterly report",
            methodology="trend analysis",
            input_data="Q3 2024 Financial Data",
            output_format="JSON"
        )
        print("Formatted prompt:")
        print(formatted_prompt)
        return True
    except Exception as e:
        print(f"Template formatting failed: {e}")
        return False

# Example 5: Performance Tracking System
class PromptPerformanceTracker:
    """
    System for tracking prompt performance metrics
    """
    def __init__(self):
        self.metrics = {
            "accuracy": [],
            "response_time": [],
            "token_usage": [],
            "user_satisfaction": []
        }
    
    def log_performance(self, accuracy: float, response_time: float, 
                       token_usage: int, user_satisfaction: float):
        self.metrics["accuracy"].append(accuracy)
        self.metrics["response_time"].append(response_time)
        self.metrics["token_usage"].append(token_usage)
        self.metrics["user_satisfaction"].append(user_satisfaction)
    
    def get_performance_summary(self) -> Dict[str, float]:
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[f"{metric}_avg"] = sum(values) / len(values)
                summary[f"{metric}_count"] = len(values)
            else:
                summary[f"{metric}_avg"] = 0.0
                summary[f"{metric}_count"] = 0
        return summary
    
    def generate_optimization_suggestions(self) -> List[str]:
        suggestions = []
        
        # Check recent accuracy
        recent_accuracy = self.metrics["accuracy"][-10:] if self.metrics["accuracy"] else []
        if recent_accuracy and sum(recent_accuracy) / len(recent_accuracy) < 0.8:
            suggestions.append("Consider refining prompt clarity and specificity")
        
        # Check token usage
        recent_tokens = self.metrics["token_usage"][-10:] if self.metrics["token_usage"] else []
        if recent_tokens and sum(recent_tokens) / len(recent_tokens) > 1000:
            suggestions.append("Optimize prompt length to reduce token usage")
        
        return suggestions

def performance_tracking_example():
    """
    Example of using the PromptPerformanceTracker
    """
    print("=== Performance Tracking Example ===")
    
    tracker = PromptPerformanceTracker()
    
    # Simulate some performance data
    tracker.log_performance(0.95, 1.2, 850, 4.5)
    tracker.log_performance(0.87, 1.5, 920, 4.2)
    tracker.log_performance(0.92, 1.1, 780, 4.7)
    
    summary = tracker.get_performance_summary()
    suggestions = tracker.generate_optimization_suggestions()
    
    print("Performance Summary:")
    for metric, value in summary.items():
        print(f"  {metric}: {value}")
    
    print("\nOptimization Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    
    return tracker

# Example 6: Prompt Evaluator
class PromptEvaluator:
    """
    System for evaluating prompt quality across multiple dimensions
    """
    def __init__(self):
        self.evaluation_criteria = {
            "clarity": {"weight": 0.25, "description": "How clear and unambiguous the prompt is"},
            "specificity": {"weight": 0.20, "description": "Level of detail and precision"},
            "completeness": {"weight": 0.20, "description": "Coverage of all necessary instructions"},
            "efficiency": {"weight": 0.15, "description": "Token efficiency and conciseness"},
            "robustness": {"weight": 0.20, "description": "Handling of edge cases and errors"}
        }
    
    def evaluate_prompt(self, prompt_text: str, test_results: Dict[str, float]) -> Dict[str, Any]:
        scores = {}
        
        # Simulate evaluation (in real implementation, this would use actual metrics)
        for criterion, config in self.evaluation_criteria.items():
            # This would be replaced with actual evaluation logic
            score = test_results.get(criterion, 0.8)  # Default score
            scores[criterion] = score
        
        # Calculate weighted total
        total_score = sum(
            scores[criterion] * config["weight"] 
            for criterion, config in self.evaluation_criteria.items()
        )
        
        return {
            "individual_scores": scores,
            "weighted_total": total_score,
            "recommendations": self._generate_improvement_suggestions(scores)
        }
    
    def _generate_improvement_suggestions(self, scores: Dict[str, float]) -> List[str]:
        suggestions = []
        
        for criterion, score in scores.items():
            if score < 0.7:
                suggestions.append(f"Improve {criterion}: {self.evaluation_criteria[criterion]['description']}")
        
        return suggestions

def prompt_evaluation_example():
    """
    Example of using the PromptEvaluator
    """
    print("=== Prompt Evaluation Example ===")
    
    evaluator = PromptEvaluator()
    
    # Test prompt
    test_prompt = """
    Extract product information from the text.
    Return as JSON.
    """
    
    # Simulate test results
    test_results = {
        "clarity": 0.6,  # Low - prompt is vague
        "specificity": 0.5,  # Low - lacks detail
        "completeness": 0.4,  # Low - missing instructions
        "efficiency": 0.9,  # High - concise
        "robustness": 0.3   # Low - no error handling
    }
    
    evaluation = evaluator.evaluate_prompt(test_prompt, test_results)
    
    print(f"Prompt: {test_prompt}")
    print(f"\nOverall Score: {evaluation['weighted_total']:.2f}")
    print("\nIndividual Scores:")
    for criterion, score in evaluation['individual_scores'].items():
        print(f"  {criterion}: {score:.2f}")
    
    print("\nRecommendations:")
    for rec in evaluation['recommendations']:
        print(f"  - {rec}")
    
    return evaluation

# Main execution function
def main():
    """
    Run all examples to test functionality
    """
    print("Testing Module 1, Section 1 Examples")
    print("=" * 50)
    
    # Test basic examples
    basic_extraction_example()
    print("\n" + "="*50 + "\n")
    
    advanced_extraction_example()
    print("\n" + "="*50 + "\n")
    
    meta_prompting_example()
    print("\n" + "="*50 + "\n")
    
    # Test template system
    template_success = prompt_template_example()
    print(f"\nTemplate system test: {'PASSED' if template_success else 'FAILED'}")
    print("\n" + "="*50 + "\n")
    
    # Test performance tracking
    performance_tracking_example()
    print("\n" + "="*50 + "\n")
    
    # Test prompt evaluation
    prompt_evaluation_example()
    print("\n" + "="*50 + "\n")
    
    print("All examples completed successfully!")

if __name__ == "__main__":
    main()
