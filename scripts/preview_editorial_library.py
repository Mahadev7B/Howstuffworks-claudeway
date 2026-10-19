"""Run a local, visual-only Explore preview from an editorial JSON batch."""

import argparse
from pathlib import Path

from lesson_platform.editorial_package import load_editorial_batch
from lesson_platform.library import prepare_publication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--port", type=int, default=59589)
    args = parser.parse_args()

    entries = load_editorial_batch(args.manifest)
    cards = [{key: entry[key] for key in
              ("slug", "title", "summary", "category", "sort_order", "featured_rank")}
             for entry in entries]
    details = {entry["slug"]: {"lesson": entry["lesson"], "title": entry["title"],
                               "summary": entry["summary"], "category": entry["category"],
                               "preview": True} for entry in entries}
    thumbnails = {entry["slug"]: prepare_publication(
        entry["lesson"], cover_data_url=entry["cover_data_url"]).thumbnail
        for entry in entries}

    import app as site
    site.app.config["EDITORIAL_PREVIEW"] = True
    site.library.list_published_cards = lambda: cards
    site.library.get_published_lesson = lambda slug: details.get(slug)
    site.library.get_published_thumbnail = lambda slug: (
        (thumbnails[slug], "image/webp") if slug in thumbnails else None)
    site.library.get_published_audio = lambda slug, slide_index: None
    site.app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
