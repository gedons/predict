# app/services/context_builder.py
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

def _format_probability(prob: float) -> str:
    """Format probability as percentage with proper handling"""
    if prob is None:
        return "N/A"
    try:
        return f"{prob:.3f} ({prob*100:.1f}%)"
    except (TypeError, ValueError):
        return "N/A"

def _format_recent_stats(recent_stats: Optional[Dict[str, Any]]) -> str:
    """Format recent statistics with better handling of missing data"""
    if not recent_stats:
        return "No recent statistics available."
    
    formatted_parts = []
    for key, value in recent_stats.items():
        if value is not None:
            # Format numeric values appropriately
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            formatted_parts.append(f"- {key}: {formatted_value}")
    
    return "\n".join(formatted_parts) if formatted_parts else "No valid statistics available."

def _get_prediction_verdict(probs: Dict[str, float]) -> str:
    """Determine the most likely outcome"""
    if not probs:
        return "Unable to determine verdict"
    
    try:
        max_outcome = max(probs.keys(), key=lambda k: probs.get(k, 0))
        max_prob = probs.get(max_outcome, 0)
        
        if max_prob > 0.5:
            confidence = "strong"
        elif max_prob > 0.4:
            confidence = "moderate"
        else:
            confidence = "weak"
        
        outcome_map = {
            "home": "home win",
            "away": "away win", 
            "draw": "draw"
        }
        
        outcome_text = outcome_map.get(max_outcome, max_outcome)
        return f"{confidence} {outcome_text} ({max_prob*100:.1f}%)"
        
    except Exception as e:
        logger.error(f"Error determining verdict: {e}")
        return "Unable to determine verdict"

def _validate_match_info(match_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean match information"""
    validated = {}
    
    # Required fields
    validated["home_team"] = str(match_info.get("home_team", "Home Team")).strip()
    validated["away_team"] = str(match_info.get("away_team", "Away Team")).strip()
    
    # Optional fields with defaults
    validated["date"] = match_info.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
    validated["league"] = match_info.get("league", "Unknown League")
    validated["venue"] = match_info.get("venue", "Unknown Venue")
    
    # Validate date format if provided
    if validated["date"]:
        try:
            datetime.strptime(validated["date"], "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format: {validated['date']}, using current date")
            validated["date"] = datetime.utcnow().strftime("%Y-%m-%d")
    
    return validated

def _build_tactical_context(recent_stats: Optional[Dict[str, Any]], match_info: Dict[str, Any]) -> str:
    """Build tactical analysis context from available data"""
    if not recent_stats:
        return "Limited tactical data available for this analysis."
    
    tactical_insights = []
    
    # Analyze attacking patterns
    home_shots = recent_stats.get("Home shots (last 5)")
    away_shots = recent_stats.get("Away shots (last 5)")
    
    if home_shots is not None and away_shots is not None:
        if home_shots > away_shots * 1.2:
            tactical_insights.append(f"{match_info['home_team']} shows superior attacking output")
        elif away_shots > home_shots * 1.2:
            tactical_insights.append(f"{match_info['away_team']} demonstrates stronger offensive metrics")
    
    # Analyze goal-scoring efficiency
    home_goals = recent_stats.get("Home goals (last 5)")
    away_goals = recent_stats.get("Away goals (last 5)")
    
    if home_goals is not None and away_goals is not None:
        goal_diff = home_goals - away_goals
        if abs(goal_diff) > 0.5:
            better_team = match_info['home_team'] if goal_diff > 0 else match_info['away_team']
            tactical_insights.append(f"{better_team} shows better goal conversion recently")
    
    # Form analysis
    form_diff = recent_stats.get("Form difference")
    if form_diff is not None:
        if abs(form_diff) > 0.2:
            if form_diff > 0:
                tactical_insights.append(f"{match_info['home_team']} in significantly better form")
            else:
                tactical_insights.append(f"{match_info['away_team']} in significantly better form")
    
    return "; ".join(tactical_insights) if tactical_insights else "Teams appear evenly matched on recent form."

def build_match_context(
    model_probs: Dict[str, float],
    match_info: Dict[str, Any],
    recent_stats: Optional[Dict[str, Any]] = None,
    extra_context: Optional[str] = None,
    include_tactical_analysis: bool = True
) -> str:
    """
    Enhanced context builder with better structure and validation.
    
    Args:
        model_probs: Model probabilities {"home": 0.4, "draw": 0.3, "away": 0.3}
        match_info: Match details
        recent_stats: Recent performance statistics
        extra_context: Additional context (news, injuries, etc.)
        include_tactical_analysis: Whether to include tactical insights
        
    Returns:
        Formatted prompt for LLM analysis
    """
    try:
        # Validate inputs
        validated_match = _validate_match_info(match_info)
        
        if not model_probs or not isinstance(model_probs, dict):
            raise ValueError("model_probs must be a non-empty dictionary")
        
        # Extract match details
        home = validated_match["home_team"]
        away = validated_match["away_team"]
        date = validated_match["date"]
        league = validated_match["league"]
        venue = validated_match["venue"]
        
        # Build probability summary
        prob_lines = [
            f"- {home} win: {_format_probability(model_probs.get('home'))}", # type: ignore
            f"- Draw: {_format_probability(model_probs.get('draw'))}", # type: ignore
            f"- {away} win: {_format_probability(model_probs.get('away'))}" # type: ignore
        ]
        
        # Get model verdict
        model_verdict = _get_prediction_verdict(model_probs)
        
        # Format statistics
        stats_section = _format_recent_stats(recent_stats)
        
        # Build tactical context if requested
        tactical_context = ""
        if include_tactical_analysis:
            tactical_context = f"\nTACTICAL INSIGHTS\n{_build_tactical_context(recent_stats, validated_match)}\n"
        
        # Clean extra context
        extra_section = extra_context.strip() if extra_context else "No additional context provided."
        
        # Build comprehensive prompt
        prompt = f"""You are an expert football analyst with deep knowledge of tactical patterns, team dynamics, and match prediction. 

Analyze the statistical prediction below and provide comprehensive insights that go beyond the raw probabilities.

MATCH DETAILS
- Home: {home}
- Away: {away}
- Date: {date}
- League: {league}
- Venue: {venue}

MODEL PREDICTION
Model verdict: {model_verdict}
{chr(10).join(prob_lines)}

PERFORMANCE STATISTICS
{stats_section}{tactical_context}

ADDITIONAL CONTEXT
{extra_section}

ANALYSIS REQUIREMENTS:
Provide your analysis as a JSON object with exactly these keys:

1. "verdict": One clear sentence summarizing your prediction
2. "explanation": 2-3 sentences explaining the key factors behind this prediction
3. "risks": List 2-3 main factors that could make this prediction unreliable
4. "suggestions": List 3-4 alternative betting markets or player props to consider
5. "confidence": Your confidence level ("high", "medium", "low")
6. "confidence_reason": One sentence explaining your confidence level

IMPORTANT:
- Be specific about tactical aspects, recent form, and statistical trends
- Consider venue advantages, travel factors, and team motivation
- Identify value opportunities where the model might be over/under-estimating
- Keep responses concise but insightful
- Base analysis on provided data, acknowledge limitations when data is sparse

Respond ONLY with valid JSON matching the required structure.""".strip()
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error building match context: {e}")
        # Return a basic fallback prompt
        return f"""Analyze this football match prediction:
{match_info.get('home_team', 'Home')} vs {match_info.get('away_team', 'Away')}
Model probabilities: {model_probs}

Provide analysis as JSON with keys: verdict, explanation, risks, suggestions, confidence, confidence_reason"""

def build_simplified_context(
    model_probs: Dict[str, float],
    home_team: str,
    away_team: str,
    additional_info: Optional[str] = None
) -> str:
    """
    Simplified context builder for quick analysis.
    
    Args:
        model_probs: Model probabilities
        home_team: Home team name
        away_team: Away team name  
        additional_info: Any additional context
        
    Returns:
        Simple prompt for LLM
    """
    prob_summary = f"Home: {model_probs.get('home', 0)*100:.1f}%, Draw: {model_probs.get('draw', 0)*100:.1f}%, Away: {model_probs.get('away', 0)*100:.1f}%"
    
    context = additional_info or "No additional information provided."
    
    return f"""Quick football match analysis needed:
{home_team} vs {away_team}
Model prediction: {prob_summary}
Context: {context}

Provide brief JSON analysis with keys: verdict, key_factors, confidence_level"""

def validate_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean LLM response to ensure required fields.
    
    Args:
        response: Raw LLM response dict
        
    Returns:
        Validated response with required fields
    """
    required_fields = {
        "verdict": "Analysis verdict not provided",
        "explanation": "No explanation available", 
        "risks": ["No risks identified"],
        "suggestions": ["No suggestions available"],
        "confidence": "medium",
        "confidence_reason": "No confidence assessment provided"
    }
    
    validated = {}
    
    for field, default in required_fields.items():
        value = response.get(field, default)
        
        # Special handling for lists
        if field in ["risks", "suggestions"]:
            if isinstance(value, str):
                validated[field] = [value]
            elif isinstance(value, list):
                validated[field] = [str(item) for item in value if item]
            else:
                validated[field] = [str(default[0])]
        else:
            validated[field] = str(value) if value else str(default)
    
    # Ensure confidence is valid
    valid_confidence = ["high", "medium", "low"]
    if validated["confidence"].lower() not in valid_confidence:
        validated["confidence"] = "medium"
    
    return validated