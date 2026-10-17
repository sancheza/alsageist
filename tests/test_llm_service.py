"""Unit tests for LLM Dialogue Service."""

import pytest
from services.llm_service import ConversationalLLMService, DEFAULT_SYSTEM_PROMPT


def test_llm_service_initialization():
    llm = ConversationalLLMService(gemini_api_key="fake_key")
    assert llm.system_prompt == DEFAULT_SYSTEM_PROMPT
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



def test_llm_groq_failure_falls_back_to_gemini(monkeypatch):
    """A Groq error falls through to Gemini and the reply enters history."""
    from unittest.mock import patch

    llm = ConversationalLLMService(groq_api_key="g", gemini_api_key="k")
    with patch("services.llm_service.cloud_api.groq_chat", side_effect=OSError("down")), \
         patch(
             "services.llm_service.cloud_api.gemini_generate_parts",
             return_value=[{"text": "*Grr*\nHalt!"}],
         ):
        reply = llm.generate_response("hello")

    assert reply == "Grr Halt!"
    assert llm.history[-1] == {"role": "assistant", "content": "Grr Halt!"}


def test_llm_groq_receives_timeout():
    """Groq calls always pass a timeout."""
    from unittest.mock import patch

    llm = ConversationalLLMService(groq_api_key="g")
    with patch("services.llm_service.cloud_api.groq_chat", return_value="ok") as chat:
        llm.generate_response("hello")
    assert chat.call_args.kwargs["timeout"] > 0


def test_llm_empty_groq_reply_falls_back_to_gemini():
    """An empty Groq reply is treated as a failure and Gemini answers instead."""
    from unittest.mock import patch

    llm = ConversationalLLMService(groq_api_key="g", gemini_api_key="k")
    with patch("services.llm_service.cloud_api.groq_chat", return_value=""), \
         patch("services.llm_service.cloud_api.gemini_generate_parts", return_value=[{"text": "Halt."}]):
        assert llm.generate_response("hello") == "Halt."
    assert llm.history[-1]["content"] == "Halt."


def test_llm_empty_groq_reply_without_gemini_raises():
    """With only Groq configured an empty reply is an error, not a silent empty line."""
    from unittest.mock import patch

    llm = ConversationalLLMService(groq_api_key="g")
    with patch("services.llm_service.cloud_api.groq_chat", return_value=""):
        with pytest.raises(RuntimeError, match="empty reply"):
            llm.generate_response("hello")
    assert llm.history == []


def test_llm_total_gemini_failure_returns_canned_line_outside_history():
    """The canned fallback line is spoken but never stored as an assistant turn."""
    from unittest.mock import patch

    llm = ConversationalLLMService(gemini_api_key="k", fallback_response="Try again, mortal.")
    with patch("services.llm_service.cloud_api.gemini_generate_parts", side_effect=OSError("down")):
        assert llm.generate_response("hello") == "Try again, mortal."
    assert llm.history == []


def test_default_groq_model_is_a_current_model(monkeypatch):
    """The default Groq chat model is gpt-oss-120b (llama-3.3-70b-versatile no longer exists)."""
    monkeypatch.delenv("GROQ_LLM_MODEL", raising=False)
    llm = ConversationalLLMService(groq_api_key="g")
    assert llm.groq_model == "openai/gpt-oss-120b"


def test_groq_request_asks_gpt_oss_for_low_reasoning():
    """gpt-oss requests carry reasoning_effort=low so max_tokens is not spent on hidden reasoning."""
    from unittest.mock import patch

    llm = ConversationalLLMService(groq_api_key="g", groq_model="openai/gpt-oss-120b")
    with patch("services.llm_service.cloud_api.groq_chat", return_value="ok") as chat:
        llm.generate_response("hello")
    assert chat.call_args.args[1]["reasoning_effort"] == "low"


def test_groq_request_omits_reasoning_effort_for_other_models():
    """Other models do not receive the gpt-oss-only field."""
    from unittest.mock import patch

    llm = ConversationalLLMService(groq_api_key="g", groq_model="qwen/qwen3.8-27b")
    with patch("services.llm_service.cloud_api.groq_chat", return_value="ok") as chat:
        llm.generate_response("hello")
    assert "reasoning_effort" not in chat.call_args.args[1]


def test_fallback_response_can_be_a_function_called_each_time():
    """A callable fallback source is evaluated on every failure, so lines can rotate."""
    from unittest.mock import patch

    lines = iter(["one", "two", "three"])
    llm = ConversationalLLMService(gemini_api_key="k", fallback_response=lambda: next(lines))
    with patch("services.llm_service.cloud_api.gemini_generate_parts", side_effect=OSError("down")):
        assert [llm.generate_response("hi") for _ in range(3)] == ["one", "two", "three"]


def test_default_fallback_response_is_neutral():
    """With nothing configured the canned line carries no persona wording."""
    from services.llm_service import DEFAULT_FALLBACK_RESPONSE

    for word in ("mortal", "bones", "bowl", "crypt"):
        assert word not in DEFAULT_FALLBACK_RESPONSE.lower()
