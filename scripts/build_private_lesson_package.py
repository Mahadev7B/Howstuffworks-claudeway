"""One local package/check command; never generates, uploads or publishes."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'draft', 'audio-dir', 'output-dir', 'ffmpeg'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--runtime-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        runtime = (args.runtime_dir or args.output_dir.parent / 'runtime-temp').resolve()
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ['TEMP'] = os.environ['TMP'] = str(runtime)
        tempfile.tempdir = str(runtime)
        from lesson_platform.private_package import build_package
        report = build_package(manifest=args.manifest, draft=args.draft, topic=args.topic,
                               audio_dir=args.audio_dir, output_dir=args.output_dir, ffmpeg=args.ffmpeg)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError, ModuleNotFoundError) as exc:
        print(json.dumps(dict(ok=False, error=str(exc), error_type=type(exc).__name__,
                              python=sys.executable, output_dir=str(args.output_dir.resolve()))))
        return 2
    print(json.dumps({**report, 'python': sys.executable, 'runtime_dir': str(runtime)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
