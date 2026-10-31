"""Build and validate one offline upload using the existing editorial/import helpers."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from .editorial_package import load_editorial_batch
from . import library, library_import
from .private_import_protocol import fingerprint_prepared, private_slide_count


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=library_import._object,
                      parse_constant=library_import._invalid_constant)


def build_package(*, manifest, draft, topic, audio_dir, output_dir, ffmpeg):
    """No generation, DB, upload, overwrite, or mutation of input files."""
    started = time.perf_counter()
    if Path(output_dir).is_symlink():
        raise ValueError('Output directory must not be a symlink')
    manifest, draft, audio_dir, output_dir = map(lambda p: Path(p).resolve(),
                                               (manifest, draft, audio_dir, output_dir))
    ffmpeg = Path(ffmpeg).resolve(strict=True)
    if not ffmpeg.is_file():
        raise ValueError('Choose the existing FFmpeg executable')
    records = _read_json(manifest)
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError('Choose a one-lesson manifest')
    count = private_slide_count(records[0].get('slug'), records[0].get('sort_order'))
    source = draft.read_text(encoding='utf-8')
    sections = re.split(r'(?m)^## ' + re.escape(topic) + r'\s*$', source)
    if len(sections) != 2:
        raise ValueError('Draft must contain one exact topic heading')
    section = sections[1].split('\n## ', 1)[0]
    lines = re.findall(r'(?m)^(\d+)\. (.+)$', section)
    if [n for n, _ in lines] != [str(i+1) for i in range(count)]:
        raise ValueError(f'Draft topic must contain exactly {count} numbered original lines')
    originals = [line for _, line in lines]
    slides = records[0].get('slides', [])
    if not isinstance(slides, list) or len(slides) != count or not all(isinstance(s, dict) for s in slides):
        raise ValueError(f'Manifest needs {count} slide objects')
    if ([s.get('narration') for s in slides] != originals or
            [s.get('explanation') for s in slides] != originals):
        raise ValueError('Manifest must preserve all original draft lines verbatim')
    entry = load_editorial_batch(manifest)[0]
    if ([s['narration'] for s in entry['lesson']['slides']] != originals or
            [s['explanation'] for s in entry['lesson']['slides']] != originals):
        raise ValueError('Original draft text must not be normalized or rewritten')
    receipt = _read_json(audio_dir / 'receipt.json')
    if not isinstance(receipt, dict):
        raise ValueError('Audio receipt must be an object')
    metadata = receipt.get('clips', [])
    if (receipt.get('slug') != entry['slug'] or not isinstance(metadata, list) or
            len(metadata) != count or not all(isinstance(meta, dict) for meta in metadata)):
        raise ValueError(f'Audio receipt must match this lesson and exactly {count} clips')
    clips = []
    for index, slide in enumerate(entry['lesson']['slides']):
        meta = metadata[index]
        if type(meta.get('slide_index')) is not int or meta['slide_index'] != index or meta.get('lesson') != entry['slug']:
            raise ValueError('Audio receipt indices/lesson do not match')
        script = library.publication_audio_text(slide)
        source_hash = hashlib.sha256(script.encode('utf-8')).hexdigest()
        if meta.get('script') != script or meta.get('source_text_hash') != source_hash:
            raise ValueError(f'Slide {index+1} narration receipt is stale')
        if (meta.get('finite') is not True or meta.get('sample_rate') != 24000 or meta.get('channels') != 1):
            raise ValueError(f'Slide {index+1} needs the completed local audio QA receipt')
        loudness, peak = meta.get('integrated_lufs'), meta.get('true_peak_dbtp')
        if (type(loudness) not in (float, int) or not -17.2 <= loudness <= -15.5 or
                type(peak) not in (float, int) or not -10 <= peak <= -1):
            raise ValueError(f'Slide {index+1} audio QA loudness/peak is outside established limits')
        audio_path = audio_dir / 'normalized' / f'slide-{index+1:02d}-kokoro-af-heart-v1.mp3'
        if not 0 < audio_path.stat().st_size <= library_import.MAX_AUDIO_BYTES:
            raise ValueError('Normalized MP3 is missing, empty or too large')
        audio = audio_path.read_bytes()
        audio_hash = hashlib.sha256(audio).hexdigest()
        if meta.get('normalized_sha256') != audio_hash:
            raise ValueError(f'Slide {index+1} normalized audio bytes changed after QA')
        clips.append(dict(slide_index=index, script=script, source_text_hash=source_hash,
                          mp3_sha256=audio_hash, mp3=base64.b64encode(audio).decode('ascii'),
                          model=meta.get('engine'), voice=meta.get('voice'), speed=meta.get('speed')))
    raw = json.dumps(dict(entry=entry, clips=clips, player_rate=0.9),
                     separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    ready = library_import.prepare_import(raw, ffmpeg_path=ffmpeg)
    if ready.entry != entry or len(ready.clips) != count:
        raise ValueError('Final importer contract verification failed')
    report = dict(ok=True, slug=entry['slug'], slides=count, audio_clips=count,
                  packet_bytes=len(raw), packet_sha256=hashlib.sha256(raw).hexdigest(),
                  package_path=str(output_dir / 'stage-packet.json'),
                  review_fingerprint=fingerprint_prepared(ready),
                  model=library_import.MODEL, voice=library_import.VOICE,
                  source_speed=1.0, player_rate=0.9, draft_topic=topic,
                  source_hashes=[c['source_text_hash'] for c in clips],
                  audio_hashes=[c['mp3_sha256'] for c in clips])
    reused = output_dir.exists()
    if reused:
        if (not output_dir.is_dir() or output_dir.is_symlink() or
                (output_dir / 'stage-packet.json').read_bytes() != raw or
                _read_json(output_dir / 'packet-receipt.json') != report):
            raise ValueError('Output already exists with different/incomplete content; choose a new output directory')
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.private-package-', dir=output_dir.parent))
        # Publish the complete directory atomically. An I/O failure leaves staging
        # available for inspection; no existing input/output is deleted or replaced.
        for name, data in (('stage-packet.json', raw),
                           ('packet-receipt.json', json.dumps(report, indent=2).encode('utf-8'))):
            with (staging / name).open('xb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(staging, output_dir)
    return {**report, 'reused': reused, 'elapsed_seconds': round(time.perf_counter()-started, 6)}
