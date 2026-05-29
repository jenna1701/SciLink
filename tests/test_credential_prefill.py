"""
Hermetic tests for environment-variable prefill of the UI credential form.

Two tiers, no Streamlit required (resolve_prefill lives in the Streamlit-free
scilink.ui.config module):

  Tier 1 — auth.py env discovery helpers (find_env_var, find_env_var_for_model,
           get_internal_proxy_base_url) and their precedence.
  Tier 2 — config.resolve_prefill: the field-by-field resolution that the
           sidebar wraps, with emphasis on the proxy-vs-vendor SAFETY GUARD
           (the proxy key must never fill the main field without a base URL).

Env isolation: the clean_env fixture clears every variable these paths read,
and tests opt back in with monkeypatch.setenv.
"""

import pytest

from scilink import auth
from scilink.ui.config import resolve_prefill


_RELEVANT_VARS = [
    "SCILINK_API_KEY", "SCILINK_BASE_URL",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "FUTUREHOUSE_API_KEY", "MP_API_KEY", "MATERIALS_PROJECT_API_KEY",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every credential env var these code paths read."""
    for var in _RELEVANT_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ─── Tier 1: auth.py env-discovery helpers ─────────────────────────


def test_find_env_var_returns_name_and_value(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    assert auth.find_env_var("openai") == ("OPENAI_API_KEY", "sk-openai")


def test_find_env_var_none_when_unset(clean_env):
    assert auth.find_env_var("openai") is None
    assert auth.find_env_var("materials_project") is None


def test_find_env_var_precedence_first_listed_wins(clean_env):
    """google lists GEMINI_API_KEY before GOOGLE_API_KEY → GEMINI wins."""
    clean_env.setenv("GOOGLE_API_KEY", "g-google")
    assert auth.find_env_var("google") == ("GOOGLE_API_KEY", "g-google")
    clean_env.setenv("GEMINI_API_KEY", "g-gemini")
    assert auth.find_env_var("google") == ("GEMINI_API_KEY", "g-gemini")


@pytest.mark.parametrize(
    "model, env_var, provider_key",
    [
        ("claude-opus-4-6", "ANTHROPIC_API_KEY", "sk-ant"),
        ("gpt-5.4", "OPENAI_API_KEY", "sk-oai"),
        ("gemini-3.1-pro-preview", "GEMINI_API_KEY", "sk-gem"),
        ("openai/gpt-4o", "OPENAI_API_KEY", "sk-oai2"),
    ],
)
def test_find_env_var_for_model_maps_provider(clean_env, model, env_var, provider_key):
    clean_env.setenv(env_var, provider_key)
    assert auth.find_env_var_for_model(model) == (env_var, provider_key)


def test_find_env_var_for_model_unknown_or_empty(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    assert auth.find_env_var_for_model("") is None
    assert auth.find_env_var_for_model("some-local-llama") is None


def test_get_internal_proxy_base_url(clean_env):
    assert auth.get_internal_proxy_base_url() is None
    clean_env.setenv("SCILINK_BASE_URL", "https://proxy.example/v1")
    assert auth.get_internal_proxy_base_url() == "https://proxy.example/v1"
    assert auth.INTERNAL_PROXY_BASE_URL == "SCILINK_BASE_URL"


# ─── Tier 2: resolve_prefill field resolution ──────────────────────


def test_vendor_key_resolves_for_model(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    out = resolve_prefill("claude-opus-4-6")
    assert out["api_key"] == ("sk-ant", "ANTHROPIC_API_KEY")
    assert out["base_url"] == ("", None)


def test_vendor_key_matches_chosen_provider_not_others(clean_env):
    """With multiple vendor keys set, only the model's provider key is used."""
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    assert resolve_prefill("gpt-5.4")["api_key"] == ("sk-oai", "OPENAI_API_KEY")
    assert resolve_prefill("claude-opus-4-6")["api_key"] == ("sk-ant", "ANTHROPIC_API_KEY")


def test_proxy_pair_used_when_base_url_in_env(clean_env):
    """SCILINK_API_KEY + SCILINK_BASE_URL → proxy key fills the main field."""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    clean_env.setenv("SCILINK_BASE_URL", "https://proxy/v1")
    out = resolve_prefill("claude-opus-4-6")
    assert out["api_key"] == ("proxy-key", "SCILINK_API_KEY")
    assert out["base_url"] == ("https://proxy/v1", "SCILINK_BASE_URL")


def test_proxy_key_used_when_base_url_already_entered(clean_env):
    """A base URL the user already typed also enables the proxy path."""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    out = resolve_prefill("claude-opus-4-6", existing_base_url="https://typed/v1")
    assert out["api_key"] == ("proxy-key", "SCILINK_API_KEY")
    # No SCILINK_BASE_URL env → base_url field itself is not prefilled.
    assert out["base_url"] == ("", None)


def test_proxy_key_NOT_used_without_base_url_falls_back_to_vendor(clean_env):
    """SAFETY GUARD: a proxy key with no base URL anywhere must NOT fill the
    main field (vendors reject it). The vendor key is used instead."""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    out = resolve_prefill("claude-opus-4-6")  # no base url anywhere
    assert out["api_key"] == ("sk-ant", "ANTHROPIC_API_KEY")
    assert out["api_key"][0] != "proxy-key"


def test_proxy_key_without_base_url_and_no_vendor_is_empty(clean_env):
    """Proxy key, no base URL, no vendor key → main field left empty (never the
    proxy key on a vendor path)."""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    out = resolve_prefill("claude-opus-4-6")
    assert out["api_key"] == ("", None)


def test_service_keys_resolve_independently_of_model(clean_env):
    clean_env.setenv("FUTUREHOUSE_API_KEY", "fh-key")
    clean_env.setenv("MP_API_KEY", "mp-key")
    out = resolve_prefill("gpt-5.4")
    assert out["fh"] == ("fh-key", "FUTUREHOUSE_API_KEY")
    assert out["mp"] == ("mp-key", "MP_API_KEY")


def test_all_fields_empty_when_nothing_set(clean_env):
    out = resolve_prefill("claude-opus-4-6")
    assert out == {
        "api_key": ("", None),
        "base_url": ("", None),
        "fh": ("", None),
        "mp": ("", None),
    }


def test_base_url_prefilled_independently_of_proxy_key(clean_env):
    """SCILINK_BASE_URL set without SCILINK_API_KEY: base_url is still surfaced,
    and the main field falls back to the vendor key."""
    clean_env.setenv("SCILINK_BASE_URL", "https://proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    out = resolve_prefill("gpt-5.4")
    assert out["base_url"] == ("https://proxy/v1", "SCILINK_BASE_URL")
    assert out["api_key"] == ("sk-oai", "OPENAI_API_KEY")
