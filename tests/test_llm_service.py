"""Unit tests for LLM Dialogue Service."""

import pytest
from services.llm_service import ConversationalLLMService, GRIMJAW_SYSTEM_PROMPT


def test_llm_service_initialization():
    llm = ConversationalLLMService(gemini_api_key="fake_key")
    assert llm.system_prompt == GRIMJAW_SYSTEM_PROMPT
    assert len(llm.history) == 0


def test_llm_history_reset():
    llm = ConversationalLLMService(gemini_api_key="fake_key")
    llm.history.append({"role": "user", "content": "hello"})
    llm.history.append({"role": "assistant", "content": "scram"})
    assert len(llm.history) == 2

    llm.reset_history()
    assert len(llm.history) == 0


def test_llm_parameterization_and_env(monkeypatch):
    """Verify custom models, prompts, fallback responses, and env var resolution."""
    # Test explicit parameters
    llm = ConversationalLLMService(
        groq_api_key="fake_groq",
        system_prompt="Custom prompt",
        groq_model="llama-3.1-8b-instant",
        fallback_response="Custom fallback response",
    )
    assert llm.system_prompt == "Custom prompt"
    assert llm.groq_model == "llama-3.1-8b-instant"
    assert llm.fallback_response == "Custom fallback response"

    # Test environment variable resolution
    monkeypatch.setenv("GROQ_LLM_MODEL", "llama-custom-env")
    monkeypatch.setenv("LLM_FALLBACK_RESPONSE", "Env fallback")
    llm_env = ConversationalLLMService(groq_api_key="fake_groq")
    assert llm_env.groq_model == "llama-custom-env"
    assert llm_env.fallback_response == "Env fallback"

