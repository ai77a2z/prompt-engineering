"""
Module 1, Section 2: Core Prompting Techniques
==============================================

This file contains runnable examples of all core prompting techniques
from Section 2, including zero-shot, few-shot, persona patterns,
output formatting, chain-of-thought, and constraint-based prompting.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
import random

class ZeroShotPrompting:
    """
    Zero-shot prompting examples and frameworks
    """
    
    @staticmethod
    def basic_zero_shot_example():
        """Basic zero-shot email extraction"""
        prompt = """
Extract all email addresses from the following text and return them as a JSON list.

Text: "You can reach out to support@example.com or for sales, contact sales-team@example.com. Do not use admin@example.com."
"""
        print("=== Basic Zero-Shot Example ===")
        print(prompt)
        return prompt
    
    @staticmethod
    def advanced_zero_shot_example():
        """Advanced zero-shot with reasoning framework"""
        advanced_zero_shot = """
# ROLE & EXPERTISE
You are an expert data extraction specialist with knowledge of email validation patterns.

# COGNITIVE APPROACH
1. Scan text for email patterns (username@domain.extension)
2. Validate each potential email against RFC standards
3. Categorize by purpose (support, sales, admin, etc.)
4. Assess confidence level for each extraction

# TASK
Extract all email addresses from the following text.

# OUTPUT REQUIREMENTS
Return a JSON object with:
- "emails": array of valid email addresses
- "categories": purpose classification for each email
- "confidence_scores": reliability rating (0.0-1.0) for each extraction
- "validation_notes": any concerns about email validity

# ERROR HANDLING
If no valid emails found, return: {"emails": [], "message": "No valid email addresses detected"}

Text: "You can reach out to support@example.com or for sales, contact sales-team@example.com. Do not use admin@example.com."
"""
        print("=== Advanced Zero-Shot with Reasoning Framework ===")
        print(advanced_zero_shot)
        return advanced_zero_shot
    
    @staticmethod
    def optimized_zero_shot_example():
        """Performance-optimized zero-shot"""
        optimized_zero_shot = """
Extract email addresses with the following constraints:
- Must follow standard email format (user@domain.tld)
- Exclude any emails marked as "do not use"
- Include confidence assessment
- Process in order of appearance

Return format: [{"email": "address", "confidence": 0.95, "context": "purpose"}]

Text: [INPUT_TEXT]
"""
        print("=== Optimized Zero-Shot Example ===")
        print(optimized_zero_shot)
        return optimized_zero_shot

class FewShotPrompting:
    """
    Few-shot prompting examples and dynamic selection
    """
    
    @staticmethod
    def basic_few_shot_example():
        """Basic few-shot classification"""
        ticket_to_classify = "The login button is not working on the mobile app."
        
        prompt = f"""
Classify the support ticket as 'Bug', 'Feature Request', or 'Question'.

Ticket: 'How do I change my password?'
Category: Question

Ticket: 'It would be great if you added a dark mode.'
Category: Feature Request

Ticket: '{ticket_to_classify}'
Category:
"""
        print("=== Basic Few-Shot Example ===")
        print(prompt)
        print("Expected Output: Bug")
        return prompt
    
    @staticmethod
    def advanced_few_shot_example():
        """Advanced few-shot with reasoning"""
        ticket_to_classify = "The login button is not working on the mobile app."
        
        advanced_few_shot = f"""
# CLASSIFICATION TASK
Classify support tickets with reasoning and confidence scores.

# EXAMPLES WITH REASONING
Ticket: "How do I change my password?"
Reasoning: User is asking for instructions on an existing feature
Category: Question
Confidence: 0.95

Ticket: "It would be great if you added a dark mode."
Reasoning: User is suggesting a new feature that doesn't currently exist
Category: Feature Request
Confidence: 0.90

Ticket: "The app crashes when I try to upload a photo."
Reasoning: User reports unexpected behavior indicating a software defect
Category: Bug
Confidence: 0.98

# CLASSIFICATION CRITERIA
- Bug: Unexpected behavior, errors, crashes, or malfunctions
- Feature Request: Suggestions for new functionality or improvements
- Question: Requests for help, information, or clarification

# YOUR TASK
Ticket: "{ticket_to_classify}"
Reasoning:
Category:
Confidence:
"""
        print("=== Advanced Few-Shot with Reasoning ===")
        print(advanced_few_shot)
        return advanced_few_shot

class DynamicFewShotSelector:
    """
    Dynamic few-shot example selection based on similarity
    """
    
    def __init__(self, example_bank: List[Dict[str, Any]]):
        self.example_bank = example_bank
    
    def select_examples(self, input_text: str, num_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Select most relevant examples based on similarity
        Note: In production, this would use actual semantic similarity
        """
        # Simulate similarity scoring (in production, use embeddings)
        scored_examples = []
        
        for example in self.example_bank:
            # Simple keyword-based similarity simulation
            similarity = self._calculate_similarity(input_text, example['input'])
            scored_examples.append((similarity, example))
        
        # Sort by similarity and return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for _, example in scored_examples[:num_examples]]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation (placeholder for real implementation)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def generate_dynamic_prompt(self, input_text: str, task_description: str) -> str:
        """Generate a prompt with dynamically selected examples"""
        selected_examples = self.select_examples(input_text)
        
        prompt = f"# TASK\n{task_description}\n\n# EXAMPLES\n"
        
        for i, example in enumerate(selected_examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n"
        
        prompt += f"\n# YOUR TASK\nInput: {input_text}\nOutput:"
        
        return prompt

def dynamic_few_shot_example():
    """Demonstrate dynamic few-shot selection"""
    print("=== Dynamic Few-Shot Selection Example ===")
    
    # Example bank for support ticket classification
    example_bank = [
        {"input": "How do I reset my password?", "output": "Question"},
        {"input": "The app crashes on startup", "output": "Bug"},
        {"input": "Please add dark mode", "output": "Feature Request"},
        {"input": "Login not working", "output": "Bug"},
        {"input": "How to export data?", "output": "Question"},
        {"input": "Need better search functionality", "output": "Feature Request"}
    ]
    
    selector = DynamicFewShotSelector(example_bank)
    
    # Test input
    test_input = "The login button doesn't respond when clicked"
    
    # Generate dynamic prompt
    dynamic_prompt = selector.generate_dynamic_prompt(
        test_input, 
        "Classify support tickets as 'Bug', 'Feature Request', or 'Question'"
    )
    
    print(dynamic_prompt)
    return dynamic_prompt

class PersonaPatterns:
    """
    Advanced persona pattern implementations
    """
    
    @staticmethod
    def basic_persona_example():
        """Basic persona pattern"""
        basic_persona = """
You are a senior financial analyst with 10+ years of experience in risk assessment.
Analyze the following investment proposal and provide your professional recommendation.

Investment Proposal: [PROPOSAL_TEXT]
"""
        print("=== Basic Persona Pattern ===")
        print(basic_persona)
        return basic_persona
    
    @staticmethod
    def multi_layered_persona_example():
        """Multi-layered persona with detailed background"""
        multi_layered_persona = """
# PROFESSIONAL IDENTITY
You are Dr. Sarah Chen, a GDPR compliance specialist with the following background:

# EXPERTISE & CREDENTIALS
- JD in Privacy Law from Stanford (2015)
- 8+ years specializing in EU data protection regulations
- Certified Information Privacy Professional (CIPP/E)
- Former legal counsel at major tech companies
- Track record of successful privacy audits

# COGNITIVE APPROACH
1. Systematic PII identification using GDPR Article 4 definitions
2. Risk assessment based on data sensitivity levels
3. Compliance gap analysis
4. Actionable remediation recommendations

# COMMUNICATION STYLE
- Precise legal terminology when appropriate
- Clear explanations for non-legal stakeholders
- Risk-based prioritization of findings
- Practical implementation guidance

# ANALYSIS FRAMEWORK
For each data element, assess:
- PII classification (personal/sensitive/special category)
- Sensitivity level (low/medium/high)
- Legal basis for processing
- Retention requirements
- Recommended actions

Message to analyze: "{message}"

Provide comprehensive GDPR compliance analysis.
"""
        print("=== Multi-Layered Persona Pattern ===")
        print(multi_layered_persona)
        return multi_layered_persona

class PersonaManager:
    """
    Dynamic persona management system
    """
    
    def __init__(self):
        self.personas = {
            "technical_expert": {
                "identity": "Senior Software Architect with 15+ years experience",
                "expertise": ["system design", "scalability", "performance optimization"],
                "communication_style": "technical precision with practical recommendations"
            },
            "business_analyst": {
                "identity": "Strategic Business Analyst with MBA and consulting background",
                "expertise": ["market analysis", "ROI calculation", "strategic planning"],
                "communication_style": "data-driven insights with executive-level communication"
            },
            "ux_researcher": {
                "identity": "Senior UX Researcher with psychology background",
                "expertise": ["user empathy", "journey mapping", "satisfaction metrics"],
                "communication_style": "empathetic and user-focused recommendations"
            }
        }
    
    def select_persona(self, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate persona based on task and context"""
        if "technical" in task_type.lower():
            return self.personas["technical_expert"]
        elif "business" in task_type.lower() or "strategy" in task_type.lower():
            return self.personas["business_analyst"]
        elif "user" in task_type.lower() or "experience" in task_type.lower():
            return self.personas["ux_researcher"]
        else:
            return self.personas["business_analyst"]  # Default
    
    def generate_persona_prompt(self, task_description: str, task_type: str, context: Dict[str, Any] = None) -> str:
        """Generate a prompt with appropriate persona"""
        persona = self.select_persona(task_type, context or {})
        
        prompt = f"""
        # ROLE & IDENTITY
        You are a {persona['identity']}.
        
        # EXPERTISE AREAS
        Your core competencies include: {', '.join(persona['expertise'])}
        
        # COMMUNICATION APPROACH
        {persona['communication_style']}
        
        # TASK
        {task_description}
        
        Provide analysis from your expert perspective.
        """
        
        return prompt

def persona_management_example():
    """Demonstrate dynamic persona management"""
    print("=== Dynamic Persona Management Example ===")
    
    manager = PersonaManager()
    
    # Test different task types
    tasks = [
        ("Analyze system performance bottlenecks", "technical analysis"),
        ("Evaluate market opportunity for new product", "business strategy"),
        ("Assess user satisfaction with current interface", "user experience")
    ]
    
    for task_desc, task_type in tasks:
        prompt = manager.generate_persona_prompt(task_desc, task_type)
        print(f"\nTask Type: {task_type}")
        print(f"Generated Prompt:\n{prompt}")
        print("-" * 50)

class OutputFormatting:
    """
    Advanced output formatting techniques
    """
    
    @staticmethod
    def basic_structured_output():
        """Basic JSON output formatting"""
        basic_format = """
Extract product information from the invoice and return as JSON.

Required format:
{
  "items": [
    {
      "name": "product name",
      "quantity": number,
      "unit_price": number,
      "total_price": number
    }
  ],
  "subtotal": number
}

Invoice text: "{invoice_text}"
"""
        print("=== Basic Structured Output ===")
        print(basic_format)
        return basic_format
    
    @staticmethod
    def schema_driven_output():
        """Schema-driven output with validation"""
        schema_driven = """
# EXTRACTION TASK
Extract invoice data following the exact schema below.

# REQUIRED JSON SCHEMA
{
  "invoice_data": {
    "items": [
      {
        "name": "string (required)",
        "quantity": "positive integer (required)",
        "unit_price": "positive number (required)",
        "total_price": "number (calculated: quantity × unit_price)"
      }
    ],
    "subtotal": "number (sum of all total_prices)",
    "metadata": {
      "extraction_confidence": "number (0.0-1.0)",
      "items_found": "integer"
    }
  }
}

# VALIDATION RULES
- All prices must be positive numbers
- Quantities must be positive integers
- Total price = quantity × unit_price
- Subtotal = sum of all total_prices

# ERROR HANDLING
If data cannot be extracted, return:
{
  "error": "extraction_failed",
  "reason": "specific reason for failure"
}

Invoice text: "{invoice_text}"
"""
        print("=== Schema-Driven Output ===")
        print(schema_driven)
        return schema_driven

class OutputFormatter:
    """
    Multi-format output support system
    """
    
    def __init__(self):
        self.formats = {
            "json": self._json_format,
            "xml": self._xml_format,
            "csv": self._csv_format,
            "markdown": self._markdown_format
        }
    
    def _json_format(self) -> str:
        return """
        # JSON OUTPUT REQUIREMENTS
        - Valid JSON syntax with proper escaping
        - Consistent key naming (snake_case)
        - Appropriate data types (string, number, boolean, array, object)
        - No trailing commas
        """
    
    def _xml_format(self) -> str:
        return """
        # XML OUTPUT REQUIREMENTS
        - Well-formed XML with proper closing tags
        - Proper element nesting
        - Attribute usage for metadata
        - CDATA sections for text content
        - Schema validation compatibility
        """
    
    def _csv_format(self) -> str:
        return """
        # CSV OUTPUT REQUIREMENTS
        - Header row with column names
        - Consistent delimiter usage (comma)
        - Proper quoting for text containing commas
        - UTF-8 encoding
        """
    
    def _markdown_format(self) -> str:
        return """
        # MARKDOWN OUTPUT REQUIREMENTS
        - Proper heading hierarchy (# ## ###)
        - Table formatting with alignment
        - Code blocks with language specification
        - Link formatting [text](url)
        """
    
    def generate_format_prompt(self, desired_format: str, task_description: str) -> str:
        """Generate format-specific prompt"""
        if desired_format not in self.formats:
            raise ValueError(f"Unsupported format: {desired_format}")
        
        format_requirements = self.formats[desired_format]()
        
        return f"""
        # TASK
        {task_description}
        
        # OUTPUT FORMAT: {desired_format.upper()}
        {format_requirements}
        
        Provide output in the specified format only.
        """

def output_formatting_example():
    """Demonstrate multi-format output support"""
    print("=== Multi-Format Output Support ===")
    
    formatter = OutputFormatter()
    task = "Extract customer information from the support ticket"
    
    for format_type in ["json", "xml", "csv", "markdown"]:
        prompt = formatter.generate_format_prompt(format_type, task)
        print(f"\n{format_type.upper()} Format Prompt:")
        print(prompt)
        print("-" * 50)

class ChainOfThoughtPrompting:
    """
    Chain-of-thought prompting implementations
    """
    
    @staticmethod
    def basic_chain_of_thought():
        """Basic chain-of-thought example"""
        incident_report = "Database connection timeout at 14:30, followed by user login failures"
        
        prompt = f"""
Analyze the following incident report. First, think step-by-step to explain the sequence of events and their relationships.
After your step-by-step analysis, conclude with the most likely root cause.

Report:
---
{incident_report}
---

Step-by-step analysis:
"""
        print("=== Basic Chain-of-Thought ===")
        print(prompt)
        return prompt
    
    @staticmethod
    def structured_chain_of_thought():
        """Structured chain-of-thought framework"""
        structured_cot = """
# SYSTEMATIC INCIDENT ANALYSIS FRAMEWORK

## STEP 1: TIMELINE RECONSTRUCTION
Identify and sequence all events mentioned:
- [List events with timestamps]

## STEP 2: DEPENDENCY MAPPING
Analyze relationships between events:
- [Map cause-and-effect relationships]

## STEP 3: IMPACT ASSESSMENT
Evaluate the scope and severity:
- [Assess user impact and system effects]

## STEP 4: ROOT CAUSE HYPOTHESIS
Based on the analysis above:
- [Primary hypothesis with supporting evidence]
- [Alternative hypotheses if applicable]

## STEP 5: VERIFICATION QUESTIONS
What additional information would confirm the root cause:
- [List specific questions or data needed]

Incident Report: {incident_report}
"""
        print("=== Structured Chain-of-Thought Framework ===")
        print(structured_cot)
        return structured_cot
    
    @staticmethod
    def multi_perspective_analysis():
        """Multi-perspective chain-of-thought"""
        multi_perspective = """
# MULTI-PERSPECTIVE INCIDENT ANALYSIS

Analyze this incident from three different expert perspectives:

## PERSPECTIVE 1: NETWORK ENGINEER
Focus on: Network connectivity, latency, infrastructure issues
Analysis:
[Infrastructure perspective analysis]

## PERSPECTIVE 2: DATABASE ADMINISTRATOR  
Focus on: Database connections, configuration changes, data integrity
Analysis:
[Database perspective analysis]

## PERSPECTIVE 3: SITE RELIABILITY ENGINEER
Focus on: System monitoring, performance metrics, service dependencies
Analysis:
[SRE perspective analysis]

## SYNTHESIS
Combine insights from all perspectives:
[Integrated analysis and recommendations]

Incident: {incident_report}
"""
        print("=== Multi-Perspective Chain-of-Thought ===")
        print(multi_perspective)
        return multi_perspective

def chain_of_thought_examples():
    """Demonstrate chain-of-thought techniques"""
    print("=== Chain-of-Thought Examples ===")
    
    cot = ChainOfThoughtPrompting()
    
    # Basic example
    cot.basic_chain_of_thought()
    print("\n" + "="*50 + "\n")
    
    # Structured example
    cot.structured_chain_of_thought()
    print("\n" + "="*50 + "\n")
    
    # Multi-perspective example
    cot.multi_perspective_analysis()

def main():
    """
    Run all core prompting technique examples
    """
    print("Testing Module 1, Section 2: Core Prompting Techniques")
    print("=" * 60)
    
    # Zero-shot examples
    zero_shot = ZeroShotPrompting()
    zero_shot.basic_zero_shot_example()
    print("\n" + "="*50 + "\n")
    
    zero_shot.advanced_zero_shot_example()
    print("\n" + "="*50 + "\n")
    
    zero_shot.optimized_zero_shot_example()
    print("\n" + "="*50 + "\n")
    
    # Few-shot examples
    few_shot = FewShotPrompting()
    few_shot.basic_few_shot_example()
    print("\n" + "="*50 + "\n")
    
    few_shot.advanced_few_shot_example()
    print("\n" + "="*50 + "\n")
    
    # Dynamic few-shot
    dynamic_few_shot_example()
    print("\n" + "="*50 + "\n")
    
    # Persona patterns
    PersonaPatterns.basic_persona_example()
    print("\n" + "="*50 + "\n")
    
    PersonaPatterns.multi_layered_persona_example()
    print("\n" + "="*50 + "\n")
    
    # Dynamic persona management
    persona_management_example()
    print("\n" + "="*50 + "\n")
    
    # Output formatting
    OutputFormatting.basic_structured_output()
    print("\n" + "="*50 + "\n")
    
    OutputFormatting.schema_driven_output()
    print("\n" + "="*50 + "\n")
    
    # Multi-format output
    output_formatting_example()
    print("\n" + "="*50 + "\n")
    
    # Chain-of-thought examples
    chain_of_thought_examples()
    
    print("\n" + "="*60)
    print("All core prompting technique examples completed successfully!")

if __name__ == "__main__":
    main()
