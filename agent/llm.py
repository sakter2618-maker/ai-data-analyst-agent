"""
llm.py — one small interface for the AI model, with three interchangeable providers.

    Gemini  (default)  needs GEMINI_API_KEY      model: GEMINI_MODEL  (default gemini-3.6-flash)
    Claude             needs ANTHROPIC_API_KEY   model: CLAUDE_MODEL  (default claude-haiku-4-5)
    Mock               no key needed — rule-based, so the agent and its tests work offline

Pick one with AI_PROVIDER=gemini|claude|mock. With no key the agent falls back to Mock
and says so in the app, instead of failing. API keys are read from environment
variables (or Streamlit secrets in the app) — never from code or GitHub.
"""

from __future__ import annotations

import json
import os
import re


class LLM:
    name = "base"

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_json(self, system: str, prompt: str, max_tokens: int = 1500) -> dict:
        text = self.complete(system + "\nReply with ONE JSON object only, no markdown.", prompt, max_tokens)
        return parse_json(text)


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


class GeminiLLM(LLM):
    name = "gemini"

    def __init__(self, api_key: str, model: str | None = None):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    def complete(self, system, prompt, max_tokens=1500):
        from google.genai import types
        r = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                               temperature=0.1))
        return r.text or ""


class ClaudeLLM(LLM):
    name = "claude"

    def __init__(self, api_key: str, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")

    def complete(self, system, prompt, max_tokens=1500):
        r = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                        messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class MockLLM(LLM):
    """Offline stand-in. It does not 'understand' language — the question → SQL step is
    handled by agent.ask.rule_based_sql — but it lets every other step run without a key."""
    name = "mock"

    def complete(self, system, prompt, max_tokens=1500):
        return json.dumps({"note": "mock"})


def get_llm(provider: str | None = None, secrets: dict | None = None) -> LLM:
    secrets = secrets or {}

    def key(name):
        return secrets.get(name) or os.getenv(name)

    provider = (provider or secrets.get("AI_PROVIDER") or os.getenv("AI_PROVIDER") or "gemini").lower()
    try:
        if provider == "gemini" and key("GEMINI_API_KEY"):
            return GeminiLLM(key("GEMINI_API_KEY"))
        if provider == "claude" and key("ANTHROPIC_API_KEY"):
            return ClaudeLLM(key("ANTHROPIC_API_KEY"))
    except Exception:  # missing SDK or bad key format -> offline mode
        pass
    return MockLLM()
