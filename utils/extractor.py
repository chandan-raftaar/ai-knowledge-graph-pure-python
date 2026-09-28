"""
extractor.py — Extract entities and relations from raw text using Claude API.
Pure urllib, no extra deps.
"""
import json
import urllib.request

SYSTEM_PROMPT = """You are a knowledge graph extraction assistant.
Given raw text, extract all named entities and the relations between them.

Return ONLY valid JSON in this exact format:
{
  "entities": [
    {"name": "Entity Name", "type": "Person|Company|Product|Place|Event|Technology|Other"}
  ],
  "relations": [
    {"from": "Entity A", "to": "Entity B", "relation": "short verb phrase"}
  ]
}

Rules:
- Entity names must match exactly as they appear in the text.
- Relation labels should be short verb phrases: founded, acquired, CEO of, works at, located in.
- Only extract relations explicitly stated or clearly implied.
- Return ONLY the JSON object. No explanation, no markdown.
"""


def extract_graph(text: str, api_key: str) -> dict:
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Extract from:\n\n{text}"}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    raw = data["content"][0]["text"].strip()
    # Strip markdown fences if Claude wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
