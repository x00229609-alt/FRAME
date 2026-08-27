"""
Gemini-based natural language synthesis of XAI explanations.

Usage:
    from XReason.gemini_synthesiser import synthesise_explanation

    text = synthesise_explanation(
        axp_string="IF age = 35 AND income = 50000 THEN label = Denied",
        cxp_string="IF income = 75000 THEN label = Approved",
        api_key="YOUR_GEMINI_API_KEY",   # or set GEMINI_API_KEY env var
    )
"""

import os
from typing import Optional, Sequence

_PROMPT_TEMPLATE = """\
You are an AI assistant helping a person understand why an automated decision was made about them.

You have been given three formal logical inputs from an AI model:

1. FULL INSTANCE EXPLANATION (the original model input and its predicted label):
   {instance_context}

2. ABDUCTIVE EXPLANATION (why this decision was made):
   {axp}

3. CONTRASTIVE EXPLANATION(S) (what would need to change to get a different outcome):
{cxp_block}

Rewrite the explanations as a single, clear, plain-English paragraph addressed directly \
to the person affected. Avoid technical jargon. Do not describe outcomes as being either positive or negative. \
Be clear not emotional.\
Do not repeat the raw logical statements verbatim. When describing the CXP, try to relate the changes to the values \
in the original instance, e.g. if X increased, if you were younger.
"""


def synthesise_explanation(
    axp_string: str,
    cxp_string: Optional[str] = None,
    cxp_strings: Optional[Sequence[str]] = None,
    instance_explanation_string: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-3.5-flash",
    class_meanings: Optional[dict] = None,
) -> str:
    """
    Send one AXP and one or more CXPs to the Gemini API and return a
    natural-language synthesis suitable for a model subject (end user).

    :param axp_string: Abductive explanation string, e.g.
                       "IF age = 35 AND income = 50000 THEN label = Denied"
    :param cxp_string: Contrastive explanation string (backward-compatible), e.g.
                       "IF income = 75000 THEN label = Approved"
    :param cxp_strings: One or more contrastive explanation strings.
    :param instance_explanation_string:bykl
                       Full explained instance string, e.g.
                       "IF ... THEN label = Denied"
    :param api_key:    Gemini API key. Falls back to the GEMINI_API_KEY
                       environment variable if not provided.
    :param model:      Gemini model name (default: "gemini-2.0-flash").
    :return:           Synthesised plain-English explanation as a string.
    :raises ValueError: If no API key is available.
    :raises RuntimeError: If the Gemini API call fails.
    """
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "A Gemini API key is required. Pass api_key= or set the "
            "GEMINI_API_KEY environment variable."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    instance_context = instance_explanation_string.strip() if instance_explanation_string else "Not provided."
    cxp_items = []
    if cxp_strings is not None:
        cxp_items.extend([str(c).strip() for c in cxp_strings if str(c).strip()])
    if cxp_string is not None and str(cxp_string).strip():
        cxp_items.append(str(cxp_string).strip())
    # keep order while deduplicating
    cxp_items = list(dict.fromkeys(cxp_items))
    if not cxp_items:
        raise ValueError("At least one contrastive explanation is required (cxp_string or cxp_strings).")

    cxp_block = "\n".join(["   - {0}".format(cxp) for cxp in cxp_items])
    prompt = _PROMPT_TEMPLATE.format(
        instance_context=instance_context,
        axp=axp_string.strip(),
        cxp_block=cxp_block
    )

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc
