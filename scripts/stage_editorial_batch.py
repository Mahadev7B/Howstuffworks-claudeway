"""Validate a local lesson batch; optionally stage private DB drafts (never publish)."""

import argparse
from pathlib import Path

from lesson_platform.editorial_package import load_editorial_batch
from lesson_platform.library import stage_editorial_lesson


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--write-private-drafts", action="store_true",
                        help="Insert inactive snapshots into the configured database")
    args = parser.parse_args()
    entries = load_editorial_batch(args.manifest)
    for entry in entries:
        print(f"VALID {entry['slug']}: {len(entry['lesson']['slides'])} illustrated slides")
    if not args.write_private_drafts:
        print("Dry run only; database unchanged.")
        return
    for entry in entries:
        stage_editorial_lesson(entry["lesson"], entry["question"], entry["slug"],
                               entry["title"], entry["summary"], entry["category"],
                               entry["sort_order"], entry["featured_rank"],
                               entry["cover_data_url"])
        print(f"STAGED PRIVATE {entry['slug']}")
    print("No lesson was activated or made public.")


if __name__ == "__main__":
    main()
