"""Run one private lesson or approve its exact local review version; no browser."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'assets/editorial/pipeline-config.example.json')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--next', action='store_true')
    modes.add_argument('--plan', action='store_true')
    modes.add_argument('--resume', action='store_true')
    modes.add_argument('--approve', metavar='SLUG')
    modes.add_argument('--services', action='store_true')
    parser.add_argument('--fingerprint')
    parser.add_argument('--retry-upload', action='store_true')
    parser.add_argument('--no-lookahead', action='store_true')
    parser.add_argument('--lookahead', action='store_true', help='Prepare only the configured next lesson while current is ready')
    args = parser.parse_args(argv)
    if args.retry_upload and not args.resume:
        parser.error('--retry-upload requires --resume and read-only reconciliation first')
    if args.lookahead and not (args.next or args.resume):
        parser.error('--lookahead requires --next or --resume')
    try:
        from lesson_platform.make_lesson import LessonPipeline, PipelineBlocked, service_inventory
        if args.services:
            print(json.dumps(service_inventory(), ensure_ascii=False))
            return 0
        config = json.loads(args.config.read_text(encoding='utf-8'))
        for key in ('workspace_root', 'catalog', 'state_path', 'work_dir', 'ffmpeg', 'preparation_dir', 'recipe_file', 'artwork_manifest'):
            if config.get(key):
                path = Path(config[key])
                if not path.is_absolute():
                    config[key] = str((ROOT/path).resolve())
        runner = LessonPipeline(config, config_path=args.config)
        if args.plan:
            result = runner.plan()
        elif args.approve:
            if not args.fingerprint:
                parser.error('--approve requires the exact --fingerprint from the reviewed version')
            result = runner.approve(args.approve, args.fingerprint, lookahead=not args.no_lookahead)
        else:
            result = runner.run_next(resume=args.resume, retry_upload=args.retry_upload, lookahead=args.lookahead)
    except Exception as exc:
        # Provider/config/transport exceptions never echo raw bodies, key values or headers.
        code = getattr(exc, 'code', type(exc).__name__)
        print(json.dumps(dict(status='blocked', code=code)))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
