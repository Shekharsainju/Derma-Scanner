"""LLM advisory layer using Groq. Provides general skincare guidance from scan results.

This is general skincare information, NOT medical diagnosis or treatment advice.
"""
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

DISCLAIMER = (
    "This is general skincare information for educational purposes only, "
    "not medical diagnosis or treatment. For real skin concerns, please consult "
    "a qualified dermatologist or healthcare professional."
)

SYSTEM_PROMPT = (
    "You are a friendly, careful skincare information assistant inside an educational "
    "skin-analysis demo app. You provide GENERAL skincare guidance only. "
    "Rules you must always follow:\n"
    "- Never give a medical diagnosis or claim certainty about any condition.\n"
    "- Never recommend prescription medication or dosages.\n"
    "- Frame everything as general lifestyle/skincare information.\n"
    "- Keep answers concise, practical, and easy to read.\n"
    "- For anything serious or persistent, advise seeing a dermatologist.\n"
    "- Do not invent statistics or pretend to have examined the user's actual skin."
)


def _get_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _is_clear_skin(scan_context: Dict) -> bool:
    result_type = scan_context.get("result_type", "")
    severity = (scan_context.get("severity") or "").lower()
    areas = scan_context.get("areas_found", 0)
    if result_type == "NO_YOLO_DETECTION_RESNET_FALLBACK":
        return True
    if severity in {"no clear activity"} and areas == 0:
        return True
    return False


def _scan_summary_text(scan_context: Dict) -> str:
    parts = []
    if scan_context.get("primary_label"):
        parts.append(f"Predicted main pattern: {scan_context['primary_label']}")
    if scan_context.get("severity"):
        parts.append(f"Visual severity level: {scan_context['severity']}")
    if scan_context.get("areas_found") is not None:
        parts.append(f"Concern regions found: {scan_context['areas_found']}")
    if scan_context.get("consumer_summary"):
        parts.append(f"Summary: {scan_context['consumer_summary']}")
    return "\n".join(parts) if parts else "No clear scan details available."


def generate_advice(scan_context: Dict) -> Dict:
    client = _get_client()
    if client is None:
        return {
            "advice": "AI advice is unavailable because the GROQ_API_KEY is not configured.",
            "mode": "unavailable",
            "disclaimer": DISCLAIMER,
        }

    clear = _is_clear_skin(scan_context)
    summary = _scan_summary_text(scan_context)

    if clear:
        user_prompt = (
            "The skin analysis found no clear skin-condition activity (skin looks generally clear).\n\n"
            f"{summary}\n\n"
            "Give short, encouraging PREVENTATIVE skincare guidance: how to keep skin clear, "
            "common everyday triggers/habits to avoid that can cause breakouts, and a simple "
            "gentle routine. Use short bullet points. Keep it positive and practical."
        )
        mode = "preventative"
    else:
        primary = scan_context.get("primary_label", "a skin concern")
        user_prompt = (
            f"The skin analysis flagged a visual pattern most similar to: {primary}.\n\n"
            f"{summary}\n\n"
            "Give a short, general explanation of what this kind of skin pattern usually looks like, "
            "common everyday contributing factors, gentle general care tips, and what to avoid. "
            "Use short bullet points. Remind the user this is not a diagnosis."
        )
        mode = "condition"

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=600,
        )
        advice = completion.choices[0].message.content.strip()
    except Exception as exc:
        return {
            "advice": f"AI advice could not be generated right now ({exc}).",
            "mode": "error",
            "disclaimer": DISCLAIMER,
        }

    return {"advice": advice, "mode": mode, "disclaimer": DISCLAIMER}


def chat_followup(scan_context: Dict, history: List[Dict], user_message: str) -> Dict:
    client = _get_client()
    if client is None:
        return {
            "reply": "AI chat is unavailable because the GROQ_API_KEY is not configured.",
            "disclaimer": DISCLAIMER,
        }

    summary = _scan_summary_text(scan_context)
    context_msg = (
        "Context: this conversation follows a skin-analysis scan with these results:\n"
        f"{summary}\n"
        "Answer the user's follow-up question using general skincare information only."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_msg},
    ]
    for turn in history[-10:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=600,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as exc:
        return {
            "reply": f"AI chat could not respond right now ({exc}).",
            "disclaimer": DISCLAIMER,
        }

    return {"reply": reply, "disclaimer": DISCLAIMER}

def suggest_products(scan_context: Dict) -> Dict:
    """Ask the LLM for commonly-recommended product names for the detected condition.

    Returns a list of {name, brand, why} dicts. Names are used later to look up
    real product data via Open Beauty Facts.
    """
    client = _get_client()
    if client is None:
        return {"products": [], "note": "GROQ_API_KEY not configured.", "disclaimer": DISCLAIMER}

    primary = scan_context.get("primary_label", "general skin concern")
    summary = _scan_summary_text(scan_context)

    user_prompt = (
        f"For a skin pattern most similar to: {primary}.\n\n"
        f"{summary}\n\n"
        "List 5 widely-recommended, commonly-available over-the-counter skincare products "
        "that people generally find helpful for this kind of concern. Prefer well-known, "
        "accessible products. Return ONLY a JSON array, no other text, no markdown. "
        'Each item must be: {"name": "...", "brand": "...", "why": "one short sentence"}. '
        "Do not include prescription medications."
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + " When asked for JSON, return only valid JSON with no markdown fences."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        raw = completion.choices[0].message.content.strip()
    except Exception as exc:
        return {"products": [], "note": f"Could not generate suggestions ({exc}).", "disclaimer": DISCLAIMER}

    # strip accidental markdown fences, then parse JSON
    import json
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        products = json.loads(cleaned)
        if not isinstance(products, list):
            products = []
    except Exception:
        products = []

    # keep only well-formed entries
    safe = []
    for p in products:
        if isinstance(p, dict) and p.get("name"):
            safe.append({
                "name": str(p.get("name", "")),
                "brand": str(p.get("brand", "")),
                "why": str(p.get("why", "")),
            })

    return {"products": safe, "note": "General community-style suggestions, not medical advice.", "disclaimer": DISCLAIMER}