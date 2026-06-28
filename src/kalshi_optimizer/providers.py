"""Provider-agnostic LLM client for the AI analysts.

Prefers a free Gemini key, then Anthropic, then OpenAI — whichever is set. Used
only for the analyst "perspectives" feature; everything else runs without an LLM.
"""

from __future__ import annotations

import requests


def provider_name(secrets) -> str | None:
    if secrets and secrets.gemini_api_key:
        return "gemini"
    if secrets and secrets.anthropic_api_key:
        return "anthropic"
    if secrets and secrets.openai_api_key:
        return "openai"
    return None


def _gemini(prompt: str, system: str | None, key: str) -> str:
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.0-flash:generateContent?key={key}")
    text = f"{system}\n\n{prompt}" if system else prompt
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}}
    r = requests.post(url, json=body, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _anthropic(prompt: str, system: str | None, key: str) -> str:
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": "claude-haiku-4-5-20251001", "max_tokens": 500,
                            "system": system or "", "messages": [{"role": "user", "content": prompt}]},
                      timeout=30)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _openai(prompt: str, system: str | None, key: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": "gpt-4o-mini", "max_tokens": 500, "messages": msgs},
                      timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm_complete(prompt: str, system: str | None, secrets) -> str:
    name = provider_name(secrets)
    if name == "gemini":
        return _gemini(prompt, system, secrets.gemini_api_key)
    if name == "anthropic":
        return _anthropic(prompt, system, secrets.anthropic_api_key)
    if name == "openai":
        return _openai(prompt, system, secrets.openai_api_key)
    raise RuntimeError("No LLM API key set (GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY)")
