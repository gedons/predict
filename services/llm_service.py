# app/services/llm_service.py
import os
import json
import time
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Custom exceptions
class LLMServiceError(Exception):
    """Base exception for LLM service errors"""
    pass

class LLMConfigurationError(LLMServiceError):
    """Raised when LLM service is not properly configured"""
    pass

class LLMAPIError(LLMServiceError):
    """Raised when LLM API call fails"""
    pass

@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    api_key: str
    model: str = "gemini-2.0-flash"
    max_output_tokens: int = 1000
    temperature: float = 0.3
    timeout_seconds: int = 30

# Try to import Google Generative AI SDK
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
    logger.info("Google Generative AI SDK loaded successfully")
except ImportError as e:
    genai = None
    GenerationConfig = None
    GENAI_AVAILABLE = False
    logger.warning(f"Google Generative AI SDK not available: {e}")

# Configuration
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def _ensure_configured() -> LLMConfig:
    """Ensure LLM service is properly configured"""
    if not GENAI_AVAILABLE:
        raise LLMConfigurationError(
            "Google Generative AI library is not installed. "
            "Install with: pip install google-generativeai"
        )
    
    if not GEMINI_KEY:
        raise LLMConfigurationError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set your Gemini API key in .env file or environment."
        )
    
    return LLMConfig(
        api_key=GEMINI_KEY,
        model=DEFAULT_MODEL,
        max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "1000")),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        timeout_seconds=int(os.getenv("GEMINI_TIMEOUT", "30"))
    )

def parse_json_like(text: str) -> Dict[str, Any]:
    """
    Robust JSON parsing with fallbacks for LLM output.
    
    Args:
        text: Raw text from LLM that may contain JSON
        
    Returns:
        Parsed JSON dict or fallback structure
    """
    if not text or not text.strip():
        return {"text": "", "error": "Empty response"}
    
    # Clean the text
    cleaned = text.strip()
    
    # Remove code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Find start and end of code block
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if start_idx == 0:
                    start_idx = i + 1
                else:
                    end_idx = i
                    break
        
        cleaned = "\n".join(lines[start_idx:end_idx])
    
    # Try to parse as JSON
    json_parse_attempts = [
        cleaned,  # As-is
        cleaned.strip('`'),  # Remove backticks
        cleaned.strip('"\''),  # Remove quotes
    ]
    
    for attempt in json_parse_attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    
    # If JSON parsing fails, try to extract JSON-like content
    try:
        # Look for content between { and }
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        
        if start >= 0 and end > start:
            json_candidate = cleaned[start:end]
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict):
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: return text with basic structure
    return {
        "text": cleaned,
        "verdict": "Analysis available in text format",
        "explanation": cleaned[:200] + "..." if len(cleaned) > 200 else cleaned,
        "parsing_error": True
    }

def _create_generation_config(config: LLMConfig) -> Any:
    """Create generation configuration for Gemini"""
    if not GenerationConfig:
        return None
    
    return GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        top_p=0.8,
        top_k=40
    )

def call_gemini(
    prompt: str, 
    model: Optional[str] = None, 
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    retry_attempts: int = 3
) -> Tuple[str, Dict[str, Any]]: # type: ignore
    """
    Enhanced Gemini API call with retry logic and better error handling.
    
    Args:
        prompt: The prompt to send to Gemini
        model: Model name override
        max_output_tokens: Token limit override
        temperature: Temperature override
        retry_attempts: Number of retry attempts on failure
        
    Returns:
        Tuple of (raw_response_text, parsed_json_dict)
        
    Raises:
        LLMConfigurationError: If service is not configured
        LLMAPIError: If API call fails after retries
    """
    config = _ensure_configured()
    
    # Override config with parameters
    if model:
        config.model = model
    if max_output_tokens:
        config.max_output_tokens = max_output_tokens
    if temperature is not None:
        config.temperature = temperature
    
    # Configure the client
    genai.configure(api_key=config.api_key) # type: ignore
    
    last_error = None
    
    for attempt in range(retry_attempts):
        try:
            logger.info(f"Calling Gemini API (attempt {attempt + 1}/{retry_attempts})")
            
            # Create model instance
            model_instance = genai.GenerativeModel(config.model) # type: ignore
            
            # Create generation config
            generation_config = _create_generation_config(config)
            
            # Make the API call
            response = model_instance.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if hasattr(response, 'text') and response.text:
                raw_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Handle candidates structure
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    raw_text = candidate.content.parts[0].text
                else:
                    raw_text = str(candidate)
            else:
                raise LLMAPIError(f"Unexpected response structure: {type(response)}")
            
            if not raw_text:
                raise LLMAPIError("Empty response from Gemini")
            
            # Parse the response
            parsed = parse_json_like(raw_text)
            
            logger.info("Successfully generated LLM analysis")
            return raw_text, parsed
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < retry_attempts - 1:
                # Wait before retrying (exponential backoff)
                wait_time = (2 ** attempt) * 1.0
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                error_msg = f"Gemini API call failed after {retry_attempts} attempts. Last error: {str(last_error)}"
                logger.error(error_msg)
                raise LLMAPIError(error_msg) from last_error

def test_llm_connection() -> Dict[str, Any]:
    """
    Test the LLM service connection and configuration.
    
    Returns:
        Dict with connection status and details
    """
    try:
        config = _ensure_configured()
        
        # Test with a simple prompt
        test_prompt = "Respond with a JSON object containing the key 'status' with value 'ok'."
        
        start_time = time.time()
        raw_text, parsed = call_gemini(test_prompt, retry_attempts=1)
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "connected",
            "model": config.model,
            "response_time_ms": round(response_time, 2),
            "test_response": parsed.get("status", "unknown"),
            "api_key_configured": bool(config.api_key),
            "sdk_available": GENAI_AVAILABLE
        }
        
    except LLMConfigurationError as e:
        return {
            "status": "configuration_error",
            "error": str(e),
            "api_key_configured": bool(GEMINI_KEY),
            "sdk_available": GENAI_AVAILABLE
        }
    except LLMAPIError as e:
        return {
            "status": "api_error", 
            "error": str(e),
            "api_key_configured": bool(GEMINI_KEY),
            "sdk_available": GENAI_AVAILABLE
        }
    except Exception as e:
        return {
            "status": "unknown_error",
            "error": str(e),
            "api_key_configured": bool(GEMINI_KEY),
            "sdk_available": GENAI_AVAILABLE
        }

# Backward compatibility function
def call_llm_analysis(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Backward compatibility wrapper"""
    return call_gemini(prompt)