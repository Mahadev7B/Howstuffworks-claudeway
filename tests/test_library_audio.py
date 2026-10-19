"""Narration is generated once in the editorial workflow, never on GET."""

from types import SimpleNamespace

import pytest

from lesson_platform import library_audio


def _settings():
    return SimpleNamespace(openai_api_key="test-key")


def test_generates_one_clip_per_slide(monkeypatch):
    calls = []
    monkeypatch.setattr(library_audio.library, "get_staged_lesson", lambda slug: {
        "slides": [{"narration": "First."}, {"narration": "Second."}]})
    monkeypatch.setattr(library_audio.library, "get_audio_metadata", lambda slug: {})
    monkeypatch.setattr(library_audio.library, "save_publication_audio", lambda *args: calls.append(args))
    class Speech:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=b"mp3")
    monkeypatch.setattr(library_audio, "OpenAI", lambda **kwargs: SimpleNamespace(audio=SimpleNamespace(speech=Speech())))
    assert library_audio.generate_publication_audio("sky", _settings()) == 2
    assert len([c for c in calls if isinstance(c, dict)]) == 2
    assert all(c["model"] == "gpt-4o-mini-tts" and c["voice"] == "marin" and c["extra_body"]["instructions"]
               for c in calls if isinstance(c, dict))


def test_matching_clip_is_reused(monkeypatch):
    narration = "First."
    digest = library_audio.text_hash(narration)
    monkeypatch.setattr(library_audio.library, "get_staged_lesson", lambda slug: {"slides": [{"narration": narration}]})
    monkeypatch.setattr(library_audio.library, "get_audio_metadata", lambda slug: {
        0: {"model": "gpt-4o-mini-tts", "voice": "marin", "source_text_hash": digest}})
    monkeypatch.setattr(library_audio, "OpenAI", lambda **kwargs: pytest.fail("matching clip should not call API"))
    assert library_audio.generate_publication_audio("sky", _settings()) == 0


def test_api_failure_does_not_activate(monkeypatch):
    monkeypatch.setattr(library_audio.library, "get_staged_lesson", lambda slug: {"slides": [{"narration": "First."}]})
    monkeypatch.setattr(library_audio.library, "get_audio_metadata", lambda slug: {})
    monkeypatch.setattr(library_audio.library, "set_publication_active", lambda *args: pytest.fail("must not activate"))
    class Speech:
        def create(self, **kwargs):
            raise RuntimeError("provider failed")
    monkeypatch.setattr(library_audio, "OpenAI", lambda **kwargs: SimpleNamespace(audio=SimpleNamespace(speech=Speech())))
    with pytest.raises(RuntimeError, match="provider failed"):
        library_audio.generate_publication_audio("sky", _settings())
