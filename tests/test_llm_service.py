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
