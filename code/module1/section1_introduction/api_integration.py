"""
Module 1, Section 1: API Integration Example
===========================================

This file contains the complete API integration example from the course material,
with proper error handling and environment setup.
"""

import os
from typing import Dict, Any, Optional

# Handle optional dependencies gracefully
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv():
        pass  # No-op if dotenv not available

# Note: This example uses Groq API, but can be adapted for other providers
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq library not installed. Install with: pip install groq")

def setup_environment():
    """
    Setup environment variables for API access
    """
    # Load API key from a .env file for security (if dotenv available)
    if DOTENV_AVAILABLE:
        load_dotenv()
    else:
        print("Info: python-dotenv not available, reading from environment directly")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found in environment variables")
        print("For production use, set your API key:")
        print("export GROQ_API_KEY=your_api_key_here")
        print("Or create a .env file with: GROQ_API_KEY=your_api_key_here")
        print("\nUsing mock client for demonstration...")
        return MockGroqClient()
    
    if not GROQ_AVAILABLE:
        print("Groq library not available - returning mock client")
        return MockGroqClient()
    
    return Groq(api_key=api_key)

class MockGroqClient:
    """
    Mock client for testing when Groq library is not available
    """
    class MockMessage:
        def __init__(self):
            self.content = "The QuantumCharge Pro device is overheating dangerously during use, posing potential safety risks."
    
    class MockChoice:
        def __init__(self):
            self.message = MockGroqClient.MockMessage()
    
    class MockCompletion:
        def __init__(self):
            self.choices = [MockGroqClient.MockChoice()]
    
    class MockChat:
        def __init__(self):
            self.completions = self
        
        def create(self, **kwargs):
            return MockGroqClient.MockCompletion()
    
    def __init__(self):
        self.chat = self.MockChat()

def analyze_support_ticket(client, support_ticket: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Analyze a support ticket using the LLM API
    
    Args:
        client: Groq client instance
        support_ticket: The support ticket text to analyze
        temperature: Temperature parameter for the API call
    
    Returns:
        Dictionary containing the analysis results
    """
    
    # The prompt template provides clear instructions and context
    prompt_template = f"""
You are a support agent assistant.
Summarize the following customer support ticket in one sentence, focusing on the core issue.

Ticket:
---
{support_ticket}
---

Summary:
"""

    try:
        # Make the API call
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_template,
                }
            ],
            model="llama3-70b-8192",  # A powerful model available on Groq
            temperature=temperature,  # Set to 0 for deterministic, factual outputs
        )

        # Extract the resulting summary
        summary = chat_completion.choices[0].message.content
        
        return {
            "success": True,
            "summary": summary,
            "model_used": "llama3-70b-8192",
            "temperature": temperature,
            "prompt_length": len(prompt_template),
            "response_length": len(summary)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def advanced_conversation_handler():
    """
    Advanced conversation handler with context management
    """
    class ConversationHandler:
        def __init__(self, client):
            self.client = client
            self.conversation_history = []
            self.system_prompt = """
You are a helpful AI assistant specialized in customer support analysis.
Maintain context across the conversation and provide consistent, professional responses.
"""
        
        def process_message(self, user_input: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
            # Enhance user input with metadata if provided
            enhanced_input = self._enhance_user_input(user_input, metadata)
            
            # Prepare messages for API call
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": enhanced_input})
            
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model="llama3-70b-8192",
                    temperature=0.3,
                    max_tokens=1000,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                
                response_content = response.choices[0].message.content
                
                # Store conversation history
                self.conversation_history.extend([
                    {"role": "user", "content": enhanced_input},
                    {"role": "assistant", "content": response_content}
                ])
                
                return self._post_process_response(response_content)
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        
        def _enhance_user_input(self, user_input: str, metadata: Optional[Dict]) -> str:
            if not metadata:
                return user_input
            
            enhanced = f"""
# USER QUERY
{user_input}

# CONTEXT METADATA
Timestamp: {metadata.get('timestamp', 'N/A')}
User Role: {metadata.get('user_role', 'N/A')}
Priority: {metadata.get('priority', 'normal')}
Department: {metadata.get('department', 'N/A')}

# PROCESSING INSTRUCTIONS
Tailor response to user role and department context.
"""
            return enhanced
        
        def _post_process_response(self, response: str) -> Dict[str, Any]:
            # Add quality checks, formatting, etc.
            return {
                "success": True,
                "response": response,
                "confidence": self._estimate_confidence(response),
                "sources_cited": self._extract_sources(response),
                "follow_up_suggestions": self._generate_follow_ups(response)
            }
        
        def _estimate_confidence(self, response: str) -> float:
            # Simple confidence estimation based on response characteristics
            confidence = 0.8  # Base confidence
            
            # Increase confidence for longer, more detailed responses
            if len(response) > 100:
                confidence += 0.1
            
            # Decrease confidence if response contains uncertainty markers
            uncertainty_markers = ["might", "could", "possibly", "perhaps", "maybe"]
            uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
            confidence -= uncertainty_count * 0.05
            
            return max(0.0, min(1.0, confidence))
        
        def _extract_sources(self, response: str) -> list:
            # Placeholder for source extraction logic
            return []
        
        def _generate_follow_ups(self, response: str) -> list:
            # Generate contextual follow-up suggestions
            return [
                "Would you like me to analyze another ticket?",
                "Do you need more details about this issue?",
                "Should I suggest next steps for resolution?"
            ]
    
    return ConversationHandler

def main():
    """
    Main function to demonstrate API integration examples
    """
    print("API Integration Examples")
    print("=" * 50)
    
    # Setup environment and client
    client = setup_environment()
    if not client:
        print("Cannot proceed without API client")
        return
    
    # Example support ticket from the course material
    support_ticket = """
Ticket ID: 8A5622
Customer: Jane Doe
Reported: 2025-07-18
Product: QuantumCharge Pro

Details: The device gets extremely hot after about 15 minutes of charging my phone.
I tried a different cable, but the issue persists.
It feels dangerously hot to the touch. Is this a known fire hazard?
"""
    
    print("=== Basic Support Ticket Analysis ===")
    print(f"Ticket: {support_ticket}")
    
    # Analyze the ticket
    result = analyze_support_ticket(client, support_ticket)
    
    if result["success"]:
        print(f"\nSummary: {result['summary']}")
        print(f"Model: {result['model_used']}")
        print(f"Temperature: {result['temperature']}")
        print(f"Prompt length: {result['prompt_length']} characters")
        print(f"Response length: {result['response_length']} characters")
    else:
        print(f"Error: {result['error']} ({result['error_type']})")
    
    print("\n" + "=" * 50)
    
    # Demonstrate advanced conversation handler
    print("=== Advanced Conversation Handler ===")
    
    ConversationHandler = advanced_conversation_handler()
    handler = ConversationHandler(client)
    
    # Test conversation with metadata
    test_message = "Can you help me prioritize this support ticket?"
    test_metadata = {
        "timestamp": "2025-07-22T10:30:00Z",
        "user_role": "support_manager",
        "priority": "high",
        "department": "customer_support"
    }
    
    conversation_result = handler.process_message(test_message, test_metadata)
    
    if conversation_result["success"]:
        print(f"User: {test_message}")
        print(f"Assistant: {conversation_result['response']}")
        print(f"Confidence: {conversation_result['confidence']:.2f}")
        print("Follow-up suggestions:")
        for suggestion in conversation_result['follow_up_suggestions']:
            print(f"  - {suggestion}")
    else:
        print(f"Conversation error: {conversation_result['error']}")
    
    print("\n" + "=" * 50)
    print("API Integration examples completed!")

if __name__ == "__main__":
    main()
