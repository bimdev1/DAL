"""Benchmarking suite for DAL vs. GPT-2 on factual synthesis tasks."""

from typing import Dict, Any, Optional

__all__ = ["evaluate_factual_synthesis"]

def evaluate_factual_synthesis(
    prompt: str, 
    model: str = "dal", 
    max_tokens: int = 100,
    domain: Optional[str] = None,
    expected_keywords: Optional[list] = None
) -> Dict[str, Any]:
    """Evaluate a model's performance on factual synthesis.
    
    Args:
        prompt: The input prompt for factual synthesis
        model: Either 'dal' or 'gpt2'
        max_tokens: Maximum number of tokens in the response
        domain: Optional domain context (e.g., 'science', 'history')
        expected_keywords: List of expected keywords in the response
        
    Returns:
        Dictionary containing evaluation metrics and metadata
    """
    result = {
        "prompt": prompt,
        "model": model,
        "domain": domain or "unknown",
        "expected_keywords": expected_keywords or [],
        "response": "",
        "token_count": 0,
        "metrics": {}
    }
    
    try:
        if model == "dal":
            from benchmark.dal_runner import generate_dal_response
            response = generate_dal_response(prompt, max_tokens, domain=domain)
            # Update result with response fields individually
            if isinstance(response, dict):
                for key, value in response.items():
                    if key in result:
                        result[key] = value
            
        elif model == "gpt2":
            from benchmark.gpt2_runner import generate_gpt2_response
            response = generate_gpt2_response(prompt, max_tokens)
            
            # Handle different response formats
            if isinstance(response, dict):
                for key, value in response.items():
                    if key in result:
                        result[key] = value
            elif isinstance(response, str):
                result["response"] = response
                result["token_count"] = len(response.split())
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Calculate keyword matches if expected_keywords provided
        if expected_keywords and result.get("response"):
            response_text = result["response"].lower()
            matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_text]
            result["matched_keywords"] = matched_keywords
            result["keyword_match_count"] = len(matched_keywords)
            result["keyword_match_ratio"] = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
        
    except Exception as e:
        result["error"] = str(e)
        print(f"Error in evaluate_factual_synthesis: {str(e)}")
    
    return result
