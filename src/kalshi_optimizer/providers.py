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


# Each free-tier model has its own daily-request bucket, so falling back across
# models routes around an exhausted quota (a fresh key in the same Google project
# shares the same buckets). Override the first choice with GEMINI_MODEL.
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite",
                 "gemini-1.5-flash"]


def _gemini_models(key: str) -> list[str]:
    import os
    pref = os.getenv("GEMINI_MODEL", "").strip()
    models = list(GEMINI_MODELS)
    if pref:
        models = [pref] + [m for m in models if m != pref]
    return models


def _gemini_one(model: str, prompt: str, system: str | None, key: str) -> str:
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    text = f"{system}\n\n{prompt}" if system else prompt
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}}
    r = requests.post(url, json=body, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _gemini(prompt: str, system: str | None, key: str) -> str:
    last: Exception | None = None
    for model in _gemini_models(key):
        try:
            return _gemini_one(model, prompt, system, key)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code in (404, 429):   # model gone or its daily bucket is spent — try next
                last = exc
                continue
            raise
    if last:
        raise last
    raise RuntimeError("no Gemini model available")


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
    import time

    name = provider_name(secrets)
    fns = {"gemini": (_gemini, getattr(secrets, "gemini_api_key", "")),
           "anthropic": (_anthropic, getattr(secrets, "anthropic_api_key", "")),
           "openai": (_openai, getattr(secrets, "openai_api_key", ""))}
    if name not in fns:
        raise RuntimeError("No LLM API key set (GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    fn, key = fns[name]
    for attempt in range(3):
        try:
            return fn(prompt, system, key)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code in (429, 503) and attempt < 2:
                time.sleep(6 * (attempt + 1))   # rate-limit backoff
                continue
            raise
