"""Validate or privately revise one existing editorial lesson and its local clips."""

import argparse
import hashlib
import json
from pathlib import Path

from lesson_platform.editorial_package import load_editorial_batch
from lesson_platform.library import publication_audio_text, revise_editorial_lesson


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_replacement_clips(entry: dict, audio_root: Path) -> dict[int, dict]:
    replacements = {}
    lesson_dir = audio_root / entry["slug"]
    if not lesson_dir.exists():
        return replacements
    for metadata_path in sorted(lesson_dir.glob("slide-*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index = metadata.get("slide_index")
        if not isinstance(index, int) or not 0 <= index < len(entry["lesson"]["slides"]):
            raise ValueError(f"Invalid slide index in {metadata_path}")
        mp3_path = metadata_path.with_suffix(".mp3")
        if metadata.get("output") != mp3_path.name or not mp3_path.is_file():
            raise ValueError(f"Missing matching MP3 for {metadata_path}")
        actual_audio_hash = _sha256(mp3_path)
        if metadata.get("output_sha256") != actual_audio_hash:
            raise ValueError(f"Audio hash mismatch for {mp3_path}")
        script = publication_audio_text(entry["lesson"]["slides"][index])
        expected_text_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
        if metadata.get("source_text_hash") != expected_text_hash:
            raise ValueError(f"Source text hash mismatch for {mp3_path}")
        if index in replacements:
            raise ValueError(f"Multiple replacement clips for slide {index + 1}")
        replacements[index] = {
            "mp3": mp3_path.read_bytes(),
            "model": metadata["engine"],
            "voice": metadata["voice"],
            "source_text_hash": expected_text_hash,
        }
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slug")
    parser.add_argument("--audio-root", type=Path, default=Path("assets/editorial/audio"))
    parser.add_argument("--write-private-draft", action="store_true")
    args = parser.parse_args()

    entries = load_editorial_batch(args.manifest)
    entry = next((item for item in entries if item["slug"] == args.slug), None)
    if entry is None:
        raise ValueError(f"Lesson not found: {args.slug}")
    replacements = _load_replacement_clips(entry, args.audio_root)
    print(f"VALID {entry['slug']}: {len(entry['lesson']['slides'])} slides, "
          f"{len(replacements)} prepared replacement clips")
    if not args.write_private_draft:
        print("Dry run only; database unchanged.")
        return

    result = revise_editorial_lesson(
        entry["lesson"], entry["question"], entry["slug"], entry["title"],
        entry["summary"], entry["category"], entry["sort_order"],
        entry["featured_rank"], cover_data_url=entry["cover_data_url"],
        audio_replacements=replacements,
    )
    previous = "public" if result["was_active"] else "private"
    print(f"REVISED PRIVATE {entry['slug']} (previously {previous}); approval reset")
    print("No lesson was activated or approved for publication.")


if __name__ == "__main__":
    main()
