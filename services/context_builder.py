# app/services/context_builder.py
from datetime import datetime
from typing import Dict, Any, Optional

def _format_recent_stats(recent_stats: Optional[Dict[str, Any]]) -> str:
    if not recent_stats:
        return "No recent stats available."
    parts = []
    for k, v in recent_stats.items():
        parts.append(f"- {k}: {v}")
    return "\n".join(parts)

def build_match_context(
    model_probs: Dict[str, float],
    match_info: Dict[str, Any],
    recent_stats: Optional[Dict[str, Any]] = None,
    extra_context: Optional[str] = None
) -> str:
    """
    Build a concise but informative prompt for the LLM.
    - model_probs: {"home":0.4,"draw":0.3,"away":0.3}
    - match_info: {"home_team":..., "away_team":..., "date":..., "league":...}
    - recent_stats: arbitrary dict with last-X matches aggregates (optional)
    - extra_context: raw news/tactical text (optional)
    """
    home = match_info.get("home_team", "HOME")
    away = match_info.get("away_team", "AWAY")
    date = match_info.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
    league = match_info.get("league", "Unknown league")
    venue = match_info.get("venue", "Unknown venue")

    prob_lines = [
        f"- {home} win: {model_probs.get('home', 0):.3f} ({model_probs.get('home', 0)*100:.1f}%)",
        f"- Draw: {model_probs.get('draw', 0):.3f} ({model_probs.get('draw', 0)*100:.1f}%)",
        f"- {away} win: {model_probs.get('away', 0):.3f} ({model_probs.get('away', 0)*100:.1f}%)"
    ]

    recent = _format_recent_stats(recent_stats)
    extra = extra_context or "No extra context provided."

    prompt = f"""
You are a concise, expert football analyst. Use the statistical probabilities and match context below to produce:
  1) A short human-readable verdict (one sentence).
  2) A brief explanation of why the model might be predicting this (tactical/availability/form reasons).
  3) Potential risk factors that would make this prediction unreliable.
  4) Suggested alternative markets or player props (short bullet list).
  5) A short confidence adjustment recommendation (raise/keep/lower) and why.

MATCH
- Home: {home}
- Away: {away}
- Date: {date}
- League: {league}
- Venue: {venue}

MODEL PROBABILITIES
{chr(10).join(prob_lines)}

RECENT STATS (if available)
{recent}

EXTRA CONTEXT (news, injuries, notes)
{extra}

Answer in JSON with keys:
  verdict, explanation, risks, suggestions, confidence, confidence_reason

Be concise and factual.
""".strip()

    return prompt
