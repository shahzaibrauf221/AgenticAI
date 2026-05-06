from __future__ import annotations

import json
import os
import re

from groq import AsyncGroq

from agents.edit_agent.intent_classifier import _keyword_classify
from agents.edit_agent.state import EditAgentState
from shared.schemas.edit_schemas import EditIntent, INTENT_TO_TARGET

_INTENT_LINES = "\n".join(
    f"- {intent} (target={target})" for intent, target in INTENT_TO_TARGET.items()
)

_SYSTEM_PROMPT = f"""You classify edit requests for a video pipeline.
Return strict JSON with keys: intent, target, scope, parameters.
Allowed intents:
{_INTENT_LINES}
Rules:
- Keep scope narrow when user specifies scene or character.
- Use scope=all when no explicit scope is provided.
- Return JSON only.
"""


async def detect_edit_intent(state: EditAgentState) -> EditAgentState:
    query = state["raw_query"]
    logs = [f"[intent_node] query={query!r}"]
    parsed: EditIntent | None = None

    try:
        client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=400,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = EditIntent.model_validate(json.loads(raw))
        logs.append(
            f"[intent_node] llm intent={parsed.intent} target={parsed.target} scope={parsed.scope}"
        )
    except Exception as exc:
        logs.append(f"[intent_node] llm unavailable, fallback={exc}")
        intent, target, scope, parameters = _keyword_classify(query)
        parsed = EditIntent(
            intent=intent,
            target=target,
            scope=scope,
            parameters=parameters,
        )
        logs.append(
            f"[intent_node] fallback intent={parsed.intent} target={parsed.target} scope={parsed.scope}"
        )

    return {
        **state,
        "intent": parsed.intent,
        "target": parsed.target,
        "scope": parsed.scope,
        "parameters": parsed.parameters,
        "status": "planning",
        "logs": logs,
    }

