"""Editorial-time voice generation for deliberately staged public lessons."""

import hashlib

from openai import OpenAI

from lesson_platform import library


MODEL = "gpt-4o-mini-tts"
VOICE = "marin"
INSTRUCTIONS = "Speak warmly and clearly for curious children age seven and older. Sound natural, upbeat, and never babyish."


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_publication_audio(slug: str, settings) -> int:
    """Save missing or stale clips; editor must listen before activating."""
    lesson = library.get_staged_lesson(slug)
    if lesson is None:
        raise ValueError("Publication was not found")
    slides = lesson.get("slides", [])
    metadata = library.get_audio_metadata(slug)
    pending = []
    for index, slide in enumerate(slides):
        narration = library.publication_audio_text(slide)
        digest = text_hash(narration)
        old = metadata.get(index)
        if old and (old["model"], old["voice"], old["source_text_hash"]) == (MODEL, VOICE, digest):
            continue
        pending.append((index, narration, digest))
    if not pending:
        return 0
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=settings.openai_api_key)
    for index, narration, digest in pending:
        # The pinned OpenAI 1.60 client predates typed `instructions`/`marin`;
        # extra_body passes this currently documented request field through.
        response = client.audio.speech.create(model=MODEL, voice=VOICE, input=narration,
                                               extra_body={"instructions": INSTRUCTIONS},
                                               response_format="mp3")
        library.save_publication_audio(slug, index, response.content, MODEL, VOICE, digest)
    return len(pending)
