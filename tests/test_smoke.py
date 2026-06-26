"""Lightweight smoke tests — no network, no API key."""

from src import config
from src.ingest import BASE, PAGES
from src.rag import PROMPT, build_context


def test_config_defaults_sane():
    assert config.RETRIEVAL_K > 0
    assert config.MAX_TOKENS > 0
    assert config.ANTHROPIC_MODEL.startswith("claude-")


def test_pages_present():
    assert PAGES and BASE.startswith("https://")


def test_prompt_includes_context_and_question():
    msgs = PROMPT.format_messages(context="CTX", question="Q?")
    assert "CTX" in msgs[0].content and "Q?" in msgs[0].content


def test_build_context_formats_sources():
    class Doc:
        page_content = "hello"
        metadata = {"source": "http://example.com"}

    out = build_context([(Doc(), 0.9)])
    assert "http://example.com" in out and "hello" in out
