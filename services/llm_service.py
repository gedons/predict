# app/services/llm_service.py
import os
import json
from typing import Tuple

try:
    import google.generativeai as genai   
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

GEMINI_KEY = os.getenv("GEMINI_API_KEY", None)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")   

def _ensure_configured():
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai library is not installed. pip install google-generativeai")
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured in environment (.env)")

def parse_json_like(text: str):
    """
    Try to parse LLM output as JSON. If fails, return {'text': raw_text}.
    """
    try:
        
        cleaned = text.strip()
        if cleaned.startswith("```"):            
            cleaned = "\n".join(line for line in cleaned.splitlines() if not line.startswith("```"))
        return json.loads(cleaned)
    except Exception:
        return {"text": text.strip()}

def call_gemini(prompt: str, model: str = None, max_output_tokens: int = 800) -> Tuple[str, dict]: # type: ignore
    """
    Call Gemini (via google.generativeai). Returns (raw_text, parsed_json_or_empty_dict).
    Raise RuntimeError if not available or key not set.
    """
    if model is None:
        model = DEFAULT_MODEL

    _ensure_configured()

    # configure client
    genai.configure(api_key=GEMINI_KEY) # type: ignore

    # The SDK interface can change; this is a defensive call pattern.
    # If your genai version supports `generate`, adjust accordingly.
    try:
        # preferred method in some SDK versions
        resp = genai.generate(model=model, prompt=prompt, max_output_tokens=max_output_tokens) # type: ignore
        # resp may have .text or .candidates structure
        raw = ""
        if hasattr(resp, "text"):
            raw = resp.text
        elif isinstance(resp, dict):
            # some SDKs return dict-like
            raw = resp.get("candidates", [{}])[0].get("content", "") or resp.get("text", "") or str(resp)
        else:
            # fallback: try repr
            raw = str(resp)
    except Exception as e:
        # if SDK doesn't have generate, try alternative attribute names (some libs changed)
        try:
            resp = genai.create(model=model, prompt=prompt, max_output_tokens=max_output_tokens) # type: ignore
            raw = resp.get("candidates", [{}])[0].get("content", "") or resp.get("output", "")
        except Exception as e2:
            raise RuntimeError(f"Failed to call Gemini: {e} / {e2}")

    parsed = parse_json_like(raw)
    return raw, parsed
