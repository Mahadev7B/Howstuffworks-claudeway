"""Offline reviewed recipes/artwork and installed local narration only.

This module has no network or image-generation bridge. Existing assistant-created
artwork must be supplied locally and bound to its prompt and source manifest.
"""
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess

from PIL import Image, ImageChops, ImageStat
from .private_import_protocol import private_slide_count


NORMALIZATION = 'loudnorm=I=-16:TP=-1.5:LRA=11'
LIMITATIONS = [
    'Structural image checks cannot guarantee anatomy, factual accuracy, or semantic relevance; human visual review is required.',
    'Audio measurements are automated checks, not human listening; listening remains pending.',
    'Artwork is supplied as an existing local file; assistant-mediated image tooling has no standalone Python bridge here.',
]


class _AudioQualityError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__('Audio QA requires finite 24 kHz mono, established LUFS/true-peak and duration limits')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_new(path, data):
    """Exclusive durable writes preserve existing and interrupted work."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Output links are not supported')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class MediaService:
    def __init__(self, config):
        self.config = dict(config)
        workspace = Path(__file__).resolve().parents[2]
        japan = workspace / 'japan-short'
        self.paths = {
            'kokoro_python': japan / '.venv-kokoro/Scripts/python.exe',
            'kokoro_model': japan / 'kokoro-models/kokoro-v1.0.onnx',
            'kokoro_voices': japan / 'kokoro-models/voices-v1.0.bin',
            'kokoro_generator': workspace / 'Howstuffworks-claudeway-library/scripts/generate_kokoro_editorial_clip.py',
            'ffmpeg': japan / 'vendor/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe',
        }
        self.paths = {key: Path(self.config.get(key) or value).resolve() for key, value in self.paths.items()}
        self._hashes = {}

    def preflight(self):
        errors = []
        for name, path in self.paths.items():
            if not path.is_file() or path.is_symlink():
                errors.append(f'Missing or unsupported local {name}: {path}')
        return dict(ok=not errors, errors=errors, estimated_cost_usd=0,
                    outside_services=[], artwork_mode='supplied_local_files',
                    paths={k: str(v) for k, v in self.paths.items()}, limitations=LIMITATIONS[:])

    @staticmethod
    def _recipe_shape(value):
        if (not isinstance(value, dict) or set(value) != {'slides', 'source_issues'} or
                not isinstance(value['slides'], list) or len(value['slides']) != 4 or
                not isinstance(value['source_issues'], list) or len(value['source_issues']) > 12 or
                any(not isinstance(issue, str) or not 0 < len(issue) <= 1000 for issue in value['source_issues'])):
            raise ValueError('Recipe must contain four separate slide recipes and source_issues')
        for slide in value['slides']:
            if (not isinstance(slide, dict) or set(slide) != {'title', 'fun_fact', 'image_prompt'} or
                    any(not isinstance(text, str) or not text.strip() or len(text) > 1600 for text in slide.values())):
                raise ValueError('Recipe may contain only separate title, fun_fact and image_prompt fields')
        return value

    def prepare_recipe(self, entry, lines, job_dir):
        if not isinstance(lines, list) or len(lines) != 4 or any(not isinstance(s, str) or not s.strip() for s in lines):
            raise ValueError('Exactly four locked original lines are required')
        recipe_file = entry.get('recipe_file') or self.config.get('recipe_file')
        file_bytes = None
        if recipe_file:
            path = Path(recipe_file)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 30000:
                raise ValueError('needs_reviewed_recipe: choose an existing local recipe file')
            file_bytes = path.read_bytes()
        if self.config.get('require_review_cache') is True:
            try:
                review_path = Path(recipe_file).with_name('review.json')
                if review_path.is_symlink() or review_path.stat().st_size > 200000:
                    raise ValueError('Unsupported review file')
                review = _read(review_path)
                if (not isinstance(review, dict) or type(review.get('schema_version')) is not int or
                        review['schema_version'] != 1 or review.get('status') != 'passed' or
                        review.get('original_lines') != lines or
                        review.get('original_lines_sha256') != _sha(_canonical(lines)) or
                        review.get('recipe_sha256') != _sha(file_bytes) or
                        review.get('source_issues') != [] or review.get('required_corrections') != []):
                    raise ValueError('Review must pass and bind exact original lines and recipe bytes without issues')
            except (OSError, TypeError, ValueError, KeyError):
                raise ValueError('review_cache_invalid: missing, held, unresolved or stale factual review') from None
        source = dict(entry=entry, original_lines=lines, recipe_file_sha256=_sha(file_bytes) if file_bytes is not None else None)
        source_hash = _sha(_canonical(source))
        job_dir = Path(job_dir)
        target = job_dir / 'recipe.json'
        receipt_path = job_dir / 'recipe-receipt.json'
        if target.exists() or receipt_path.exists():
            try:
                receipt = _read(receipt_path)
                if receipt['source_sha256'] != source_hash or receipt['recipe_sha256'] != _sha(target.read_bytes()):
                    raise ValueError('Recipe cache source or output changed')
                return self._recipe_shape(_read(target))
            except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError('Existing recipe cache is incomplete or changed') from None
        lines_hash = _sha(_canonical(lines))
        if recipe_file:
            supplied = json.loads(file_bytes)
            if (not isinstance(supplied, dict) or supplied.get('reviewed') is not True or
                    supplied.get('original_lines_sha256') != lines_hash):
                raise ValueError('needs_reviewed_recipe: local recipe must be reviewed and bound to the exact original lines')
            recipe = self._recipe_shape(supplied.get('recipe'))
        else:
            if entry.get('recipe_reviewed') is not True or 'recipe' not in entry:
                raise ValueError('needs_reviewed_recipe: supply a reviewed local recipe; no automatic enrichment is available')
            recipe = self._recipe_shape(entry['recipe'])
        raw = _canonical(recipe)
        _write_new(target, raw)
        _write_new(receipt_path, _canonical(dict(source_sha256=source_hash, recipe_sha256=_sha(raw),
                                                original_lines_sha256=lines_hash, estimated_cost_usd=0)))
        return recipe

    def make_image(self, prompt, target, index):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 8000:
            raise ValueError('Image prompt must be nonempty and at most 8000 UTF-8 bytes')
        if type(index) is not int or not 0 <= index <= 3:
            raise ValueError('Image index must be 0 through 3')
        target = Path(target)
        if target.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'} or target.is_symlink():
            raise ValueError('Image target must be a raster file, not a link')
        prompt_hash = _sha(prompt.encode('utf-8'))
        manifest_path = self.config.get('artwork_manifest')
        if not manifest_path:
            raise ValueError('needs_builtin_artwork: supply the existing assistant-created artwork manifest')
        manifest_path = Path(manifest_path).resolve()
        if not manifest_path.is_file() or manifest_path.stat().st_size > 50000:
            raise ValueError('needs_builtin_artwork: artwork manifest is missing or too large')
        try:
            artwork = _read(manifest_path)
            source_manifest = (manifest_path.parent / artwork['source_manifest']).resolve()
            source_bytes = source_manifest.read_bytes()
            if artwork['source_manifest_sha256'] != _sha(source_bytes):
                raise ValueError('Artwork source manifest changed')
            records = json.loads(source_bytes)
            if (not isinstance(records, list) or len(records) != 1 or
                    len(records[0]['slides']) != 4 or len(artwork['slides']) != 4 or
                    records[0]['slides'][index]['image_prompt'] != prompt):
                raise ValueError('Artwork prompt does not match the source manifest')
            supplied = artwork['slides'][index]
            source_path = (manifest_path.parent / supplied['path']).resolve()
            if supplied['prompt_sha256'] != prompt_hash:
                raise ValueError('Artwork prompt binding changed')
            if not source_path.is_file() or source_path.stat().st_size > 15_000_000:
                raise ValueError('Artwork source is missing or too large')
            source_data = source_path.read_bytes()
            if supplied['sha256'] != _sha(source_data):
                raise ValueError('Artwork source bytes changed')
        except (KeyError, IndexError, OSError, TypeError, json.JSONDecodeError):
            raise ValueError('needs_builtin_artwork: supplied artwork bindings are missing or invalid') from None
        source_hash = _sha(_canonical(dict(prompt_sha256=prompt_hash, index=index,
                            source_manifest_sha256=_sha(source_bytes), artwork_sha256=_sha(source_data))))
        receipt_path = target.with_suffix('.image-receipt.json')
        if target.exists() or receipt_path.exists():
            try:
                receipt = _read(receipt_path)
                if receipt['binding_sha256'] != source_hash or receipt['sha256'] != _sha(target.read_bytes()):
                    raise ValueError('Existing image cache source or bytes changed')
                if not self.check_images([target])['ok']:
                    raise ValueError('Existing cached image failed structural checks')
                return {**receipt, 'reused': True}
            except (KeyError, OSError, TypeError, json.JSONDecodeError):
                raise ValueError('Existing image cache is incomplete or changed') from None
        report = self.check_images([source_path])['images'][0]
        if report['flags']:
            raise ValueError('needs_builtin_artwork: existing artwork failed structural checks')
        with Image.open(BytesIO(source_data)) as image:
            expected = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.webp': 'WEBP'}[target.suffix.lower()]
            if image.format != expected:
                raise ValueError('Artwork target extension must preserve the source file format')
        _write_new(target, source_data)
        receipt = dict(path=str(target), sha256=_sha(source_data), width=report['width'], height=report['height'],
                       flags=[], cost_usd=0, binding_sha256=source_hash, prompt_sha256=prompt_hash,
                       source_manifest_sha256=_sha(source_bytes), source_path=str(source_path), reused=False)
        _write_new(receipt_path, _canonical(receipt))
        return receipt

    def check_images(self, paths):
        results, decoded = [], []
        for path in paths:
            path = Path(path)
            item = dict(path=str(path), flags=[], width=None, height=None)
            results.append(item)
            try:
                if path.is_symlink() or path.stat().st_size > 15_000_000:
                    raise ValueError('Unsupported image')
                raw = path.read_bytes()
                with Image.open(BytesIO(raw)) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError('Image is too large')
                    image.load()
                    item.update(width=image.width, height=image.height, sha256=_sha(raw))
                    expected = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.webp': 'WEBP'}.get(path.suffix.lower())
                    if image.format != expected or not expected:
                        item['flags'].append('wrong_format')
                    if not (640 <= image.width <= 4096 and 400 <= image.height <= 4096):
                        item['flags'].append('wrong_dimensions')
                    if not 1.45 <= image.width / image.height <= 1.9:
                        item['flags'].append('wrong_aspect')
                    thumb = image.convert('RGB').resize((64, 40), Image.Resampling.LANCZOS)
                if max(ImageStat.Stat(thumb).stddev) < 4:
                    item['flags'].append('blank_or_near_uniform')
                for previous, other in decoded:
                    difference = ImageStat.Stat(ImageChops.difference(thumb, other)).mean
                    if item['sha256'] == previous['sha256']:
                        item['flags'].append('duplicate')
                        break
                    if sum(difference)/3 < 3:
                        item['flags'].append('near_duplicate')
                        break
                decoded.append((item, thumb))
            except (OSError, ValueError, Image.DecompressionBombError):
                item['flags'].append('decode_failed')
        return dict(ok=bool(results) and all(not item['flags'] for item in results), images=results, limitations=LIMITATIONS[:1])

    def _file_hash(self, path):
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._hashes:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(chunk)
            self._hashes[key] = digest.hexdigest()
        return self._hashes[key]

    def _run(self, args, temp_dir):
        # Set these for every child from its start, never changing the parent's
        # environment. Runtime/model imports consequently see a local temp path.
        temp_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'TEMP': str(temp_dir.resolve()), 'TMP': str(temp_dir.resolve()),
               'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
        try:
            return subprocess.run([str(a) for a in args], check=True, capture_output=True,
                                  text=True, encoding='utf-8', env=env, timeout=120)
        except (subprocess.SubprocessError, OSError):
            raise ValueError('Local audio command failed; inspect preserved local artifacts before resuming') from None

    def _audio_qa(self, target, temp_dir):
        probe = ('import soundfile as sf,numpy as np,json,sys;'
                 'a,s=sf.read(sys.argv[1]);print(json.dumps(dict(duration_seconds=len(a)/s,'
                 'sample_rate=s,channels=1 if a.ndim==1 else a.shape[1],'
                 'peak=float(np.max(np.abs(a))),finite=bool(np.isfinite(a).all()))))')
        details = json.loads(self._run([self.paths['kokoro_python'], '-c', probe, target], temp_dir).stdout)
        measured = self._run([self.paths['ffmpeg'], '-nostdin', '-hide_banner', '-i', target,
                              '-af', NORMALIZATION+':print_format=json', '-f', 'null', os.devnull], temp_dir)
        loudness = json.loads(measured.stderr[measured.stderr.rfind('{'):measured.stderr.rfind('}')+1])
        details.update(integrated_lufs=float(loudness['input_i']), true_peak_dbtp=float(loudness['input_tp']))
        try:
            self._audio_gate(details)
        except ValueError:
            raise _AudioQualityError(details) from None
        return details

    @staticmethod
    def _audio_gate(meta):
        if (meta.get('finite') is not True or meta.get('sample_rate') != 24000 or meta.get('channels') != 1 or
                any(type(meta.get(k)) not in (int, float) or not math.isfinite(meta[k]) for k in
                    ('duration_seconds', 'peak', 'integrated_lufs', 'true_peak_dbtp')) or
                not 0 < meta['peak'] < 1 or not 1 < meta['duration_seconds'] < 180 or
                not -17.2 <= meta['integrated_lufs'] <= -15.5 or not -10 <= meta['true_peak_dbtp'] <= -1):
            raise ValueError('Audio QA requires finite 24 kHz mono, established LUFS/true-peak and duration limits')

    @staticmethod
    def _two_pass_filter(measured):
        fields = [('measured_I', 'input_i'), ('measured_TP', 'input_tp'), ('measured_LRA', 'input_lra'),
                  ('measured_thresh', 'input_thresh'), ('offset', 'target_offset')]
        values = [(option, float(measured[key])) for option, key in fields]
        if any(not math.isfinite(value) for _, value in values):
            raise ValueError('Two-pass normalization requires finite raw measurements')
        return NORMALIZATION + ''.join(f':{option}={value}' for option, value in values) + ':linear=true'

    def _retry_loudness(self, raw, target, binding_path, meta, temp_dir, failed_qa):
        """One normalization-only attempt; retain failed bytes and never retry TTS."""
        lufs = failed_qa.get('integrated_lufs')
        if type(lufs) not in (int, float) or not math.isfinite(lufs) or -17.2 <= lufs <= -15.5:
            raise _AudioQualityError(failed_qa)
        # Changing only this value must satisfy every other unchanged gate.
        self._audio_gate({**failed_qa, 'integrated_lufs': -16.0})
        failed_output = target.with_suffix('.one-pass-failed.mp3')
        failed_binding = target.with_suffix('.one-pass-failed.normalization-receipt.json')
        # Exclusive backups also bound attempts across interrupted/restarted runs.
        _write_new(failed_output, target.read_bytes())
        _write_new(failed_binding, binding_path.read_bytes())
        measured = self._run([self.paths['ffmpeg'], '-nostdin', '-hide_banner', '-i', raw,
                              '-af', NORMALIZATION+':print_format=json', '-f', 'null', os.devnull], temp_dir)
        measurements = json.loads(measured.stderr[measured.stderr.rfind('{'):measured.stderr.rfind('}')+1])
        applied_filter = self._two_pass_filter(measurements)
        provenance = dict(method='measured_two_pass', filter=applied_filter, measurements=measurements,
                          raw_sha256=meta['output_sha256'], source_text_hash=meta['source_text_hash'],
                          failed_output=str(failed_output), failed_output_sha256=self._file_hash(failed_output),
                          failed_binding=str(failed_binding), failed_binding_sha256=self._file_hash(failed_binding),
                          failed_qa=failed_qa)
        _write_new(target.with_suffix('.two-pass-attempt.json'), _canonical(provenance))
        candidate = target.with_suffix('.two-pass.mp3')
        self._run([self.paths['ffmpeg'], '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', raw,
                   '-af', applied_filter, '-ar', '24000', '-ac', '1', '-codec:a', 'libmp3lame', '-q:a', '2',
                   '-n', candidate], temp_dir)
        details = self._audio_qa(candidate, temp_dir)
        provenance['repaired_qa'] = details
        binding = dict(raw_sha256=meta['output_sha256'], normalized_sha256=self._file_hash(candidate),
                       filter=applied_filter, normalization_fallback=provenance)
        candidate_binding = candidate.with_suffix('.normalization-receipt.json')
        _write_new(candidate_binding, _canonical(binding))
        # Only the failed, unreceipted clip is replaced, after complete QA and
        # exclusive preservation. Packaging retains its established filenames.
        os.replace(candidate, target)
        os.replace(candidate_binding, binding_path)
        return details, provenance

    def make_audio(self, manifest, job_dir):
        manifest, job_dir = Path(manifest).resolve(), Path(job_dir).resolve()
        entries = _read(manifest)
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
            raise ValueError('Audio requires one lesson manifest')
        entry = entries[0]
        count = private_slide_count(entry.get('slug'), entry.get('sort_order'))
        slides = entry.get('slides')
        if not isinstance(slides, list) or len(slides) != count or not isinstance(entry.get('slug'), str):
            raise ValueError(f'Audio requires exactly {count} slides and a slug')
        scripts = []
        for slide in slides:
            if (not isinstance(slide, dict) or any(not isinstance(slide.get(k), str) or not slide[k].strip()
                    for k in ('narration', 'explanation', 'fun_fact'))):
                raise ValueError('Each slide requires original narration/explanation and separate fun_fact')
            if slide['narration'] != slide['explanation'] or slide['narration'] != slide['narration'].strip():
                raise ValueError('Narration must preserve the original verbatim without boundary whitespace')
            scripts.append(slide['narration']+' Fun fact: '+slide['fun_fact'].strip())
        receipt_path = job_dir / 'receipt.json'
        manifest_hash = _sha(manifest.read_bytes())
        if receipt_path.exists():
            try:
                receipt = _read(receipt_path)
                if (receipt['source_manifest_sha256'] != manifest_hash or receipt['slug'] != entry['slug'] or
                        len(receipt['clips']) != count or receipt['player_rate'] != 0.9 or
                        receipt['normalization'] != NORMALIZATION+'; no tempo filter'):
                    raise ValueError('Audio cache is stale or changed')
                runtime_hashes = {k+'_sha256': self._file_hash(v) for k, v in self.paths.items()}
                if receipt['runtime_hashes'] != runtime_hashes:
                    raise ValueError('Audio cache runtime/model/voice changed')
                for i, (clip, script) in enumerate(zip(receipt['clips'], scripts)):
                    raw = job_dir/'raw'/f'slide-{i+1:02d}-kokoro-af-heart-v1.mp3'
                    target = job_dir/'normalized'/raw.name
                    self._audio_gate(clip)
                    if (clip['script'] != script or clip['source_text_hash'] != _sha(script.encode('utf-8')) or
                            clip['slide_index'] != i or clip['lesson'] != entry['slug'] or
                            clip['voice'] != 'af_heart' or clip['speed'] != 1.0 or
                            clip['engine'] != 'Kokoro-82M ONNX v1.0' or clip['language'] != 'en-us' or
                            clip['model_sha256'] != runtime_hashes['kokoro_model_sha256'] or
                            clip['voices_sha256'] != runtime_hashes['kokoro_voices_sha256'] or
                            clip['normalized_sha256'] != self._file_hash(target) or
                            clip['output_sha256'] != self._file_hash(raw)):
                        raise ValueError('Audio cache source or bytes changed')
                return receipt
            except (KeyError, TypeError, OSError, json.JSONDecodeError):
                raise ValueError('Existing audio cache is incomplete or changed') from None
        for name, path in self.paths.items():
            if not path.is_file() or path.is_symlink():
                raise ValueError(f'Missing supported local audio {name}')
        runtime_hashes = {k+'_sha256': self._file_hash(v) for k, v in self.paths.items()}
        temp_dir = job_dir/'temp'
        clips = []
        for i, script in enumerate(scripts):
            raw = job_dir/'raw'/f'slide-{i+1:02d}-kokoro-af-heart-v1.mp3'
            target = job_dir/'normalized'/raw.name
            source_hash = _sha(script.encode('utf-8'))
            if not raw.exists() and not raw.with_suffix('.json').exists():
                self._run([self.paths['kokoro_python'], self.paths['kokoro_generator'], manifest, entry['slug'], i+1, raw,
                           '--model', self.paths['kokoro_model'], '--voices', self.paths['kokoro_voices']], temp_dir)
            try:
                meta = _read(raw.with_suffix('.json'))
                if (meta['script'] != script or meta['source_text_hash'] != source_hash or
                        meta['output_sha256'] != self._file_hash(raw) or meta['voice'] != 'af_heart' or meta['speed'] != 1.0 or
                        meta['engine'] != 'Kokoro-82M ONNX v1.0' or meta['language'] != 'en-us' or
                        meta['model_sha256'] != runtime_hashes['kokoro_model_sha256'] or
                        meta['voices_sha256'] != runtime_hashes['kokoro_voices_sha256'] or
                        meta['lesson'] != entry['slug'] or meta['slide_index'] != i):
                    raise ValueError('Raw audio cache source or hashes changed')
            except (KeyError, TypeError, OSError, json.JSONDecodeError):
                raise ValueError('Raw audio cache is incomplete; preserve it for review') from None
            binding_path = target.with_suffix('.normalization-receipt.json')
            if not target.exists() and not binding_path.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                self._run([self.paths['ffmpeg'], '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', raw,
                           '-af', NORMALIZATION, '-ar', '24000', '-ac', '1', '-codec:a', 'libmp3lame', '-q:a', '2', '-n', target], temp_dir)
                _write_new(binding_path, _canonical(dict(raw_sha256=meta['output_sha256'],
                           normalized_sha256=self._file_hash(target), filter=NORMALIZATION)))
            try:
                binding = _read(binding_path)
                if (binding['raw_sha256'] != meta['output_sha256'] or
                        binding['normalized_sha256'] != self._file_hash(target)):
                    raise ValueError('Normalized audio source binding changed')
                if binding['filter'] != NORMALIZATION:
                    fallback = binding['normalization_fallback']
                    if (binding['filter'] != self._two_pass_filter(fallback['measurements']) or
                            fallback['raw_sha256'] != meta['output_sha256'] or
                            fallback['source_text_hash'] != source_hash or
                            fallback['failed_output_sha256'] != self._file_hash(target.with_suffix('.one-pass-failed.mp3')) or
                            fallback['failed_binding_sha256'] != self._file_hash(target.with_suffix('.one-pass-failed.normalization-receipt.json'))):
                        raise ValueError('Two-pass normalization provenance changed')
            except (KeyError, TypeError, OSError, json.JSONDecodeError):
                raise ValueError('Normalized audio binding is incomplete; preserve it for review') from None
            fallback = binding.get('normalization_fallback')
            try:
                details = self._audio_qa(target, temp_dir)
            except _AudioQualityError as exc:
                if fallback is not None:
                    raise
                details, fallback = self._retry_loudness(raw, target, binding_path, meta, temp_dir, exc.details)
            clips.append({**meta, **details, 'normalized_output': str(target),
                          'normalized_sha256': self._file_hash(target),
                          **({'normalization_fallback': fallback} if fallback is not None else {}),
                          'player_seconds_at_0_9': details['duration_seconds']/0.9})
        receipt = dict(slug=entry['slug'], clips=clips, source_manifest_sha256=manifest_hash, runtime_hashes=runtime_hashes,
                       normalization=NORMALIZATION+'; no tempo filter', player_rate=0.9,
                       human_listening='pending', total_seconds=sum(c['duration_seconds'] for c in clips))
        _write_new(receipt_path, _canonical(receipt))
        return receipt
