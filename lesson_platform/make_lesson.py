"""One bounded private lesson job, with durable review/uncertainty state."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

from .lesson_queue import LessonQueue
from .pipeline_media import MediaService
from .private_http import PrivateImportClient, UploadUncertain
from .private_import_protocol import valid_receipt
from .private_package import build_package


class PipelineBlocked(RuntimeError):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or code)


def service_inventory():
    return dict(paid_provider_apis_used=False, local_tools_have_per_call_service_fee=False,
                fully_standalone_image_generation=False,
                local_tools=['Installed Kokoro ONNX af_heart', 'Installed FFmpeg', 'Pillow and Python'],
                outside_services=[
                    dict(name='AskLilOwl website / existing Render hosting',
                         used_by='Direct HTTPS private upload only', confirmed_universally_free=False,
                         cost='Existing hosting terms apply; no new service or hosting plan is added'),
                    dict(name='Existing ChatGPT built-in image tool',
                         used_by='Assistant-mediated artwork preparation, not called by this Python command',
                         confirmed_universally_free=False,
                         cost='Plan/usage terms are not exposed to this pipeline; no separate paid API is used')])


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _immutable(path, data):
    path = Path(path)
    if path.is_symlink():
        raise PipelineBlocked('artifact_link', 'Refusing linked output')
    if path.exists():
        if path.read_bytes() != data:
            raise PipelineBlocked('locked_input_changed', 'Existing locked artifact differs; no overwrite')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class _Timing:
    def __init__(self, job_dir):
        self.path = job_dir / 'timing.jsonl'
        self.values = {}
        self.lock = threading.Lock()

    def event(self, stage, event, **data):
        item = dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage, event=event, **data)
        with self.lock, self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item, separators=(',', ':')) + '\n')
            handle.flush()

    def call(self, stage, function, *args):
        started = time.perf_counter()
        self.event(stage, 'start')
        try:
            result = function(*args)
        except Exception as exc:
            self.event(stage, 'blocked', error_type=type(exc).__name__)
            raise
        elapsed = round(time.perf_counter()-started, 6)
        with self.lock:
            self.values[stage] = elapsed
        self.event(stage, 'finish', seconds=elapsed)
        return result


class LessonPipeline:
    def __init__(self, config, *, config_path=None, media=None, client_factory=PrivateImportClient):
        self.config = dict(config)
        if set(config) & {'paid_generation_approved', 'per_lesson_budget_usd', 'recipe_enrichment', 'flux_model'}:
            raise PipelineBlocked('paid_provider_config_removed', 'Only local narration and existing built-in-tool artwork are supported')
        if any(k.lower() in {'token', 'import_token', 'api_key', 'secret', 'fal_key', 'anthropic_api_key'} for k in config):
            raise PipelineBlocked('secret_in_config', 'Credentials belong only in approved environment storage')
        self.config_path = Path(config_path).resolve() if config_path else None
        self.workspace = Path(config['workspace_root']).resolve()
        self.work_dir = Path(config['work_dir']).resolve()
        self.state_path = Path(config['state_path']).resolve()
        self.execution_lock = self.state_path.with_suffix('.run.lock')
        self.queue = LessonQueue(config['catalog'], self.state_path)
        self._media_override = media
        self.media = media or MediaService(config)
        self.client_factory = client_factory

    def _blockers(self, needs_media=True, check_credentials=True):
        result = []
        if self.config.get('upload_authorized') is not True:
            result.append('dedicated_upload_access_approval_and_user_setup_required')
        if self.config.get('server_url') not in {'https://www.asklilowl.com', 'https://asklilowl.com'}:
            result.append('official_https_origin_required')
        if needs_media:
            report = self.media.preflight()
            if not report.get('ok'):
                result.append('local_media_preflight_failed')
        # Never inspect credentials before the specific permission gates are met.
        if not result and check_credentials:
            if not os.environ.get('ASKLILOWL_IMPORT_TOKEN'):
                result.append('dedicated_import_token_environment_missing')
        return result

    def plan(self):
        snapshot = self.queue.snapshot()
        return dict(status='plan_only', next=self.queue.next_entry(), current=snapshot['current'],
                    blockers=self._blockers(check_credentials=False), generation_performed=False,
                    upload_performed=False, credential_presence_checked=False, services=service_inventory())

    @contextmanager
    def _execution(self):
        self.execution_lock.parent.mkdir(parents=True, exist_ok=True)
        owner = uuid.uuid4().hex.encode('ascii')
        try:
            fd = os.open(self.execution_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise PipelineBlocked('runner_busy', 'Another run owns the execution lock; do not reclaim it automatically') from None
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(owner)
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            if self.execution_lock.read_bytes() != owner:
                raise PipelineBlocked('execution_lock_changed', 'Lock ownership changed; preserved for inspection')
            self.execution_lock.unlink()

    def _client(self):
        return self.client_factory(self.config['server_url'], os.environ['ASKLILOWL_IMPORT_TOKEN'])

    def _job_media(self, entry):
        if self._media_override is not None:
            return self._media_override
        selected = dict(self.config)
        if selected.get('preparation_dir'):
            folder = Path(selected['preparation_dir']).resolve() / entry['slug']
            selected['recipe_file'] = selected.get('recipe_file') or str(folder/'recipe.json')
            selected['artwork_manifest'] = selected.get('artwork_manifest') or str(folder/'artwork.json')
        return MediaService(selected)

    def _ready(self, job):
        return dict(status='ready', slug=job['entry']['slug'], preview_url=job['preview_url'],
                    job_dir=job['job_dir'], review_fingerprint=job['review_fingerprint'],
                    timings=job.get('timings', {}), slowest_stage=job.get('slowest_stage'),
                    private=True, approved=False, artwork_mode='existing_builtin_tool_files',
                    timings_include_artwork_creation=False,
                    timing_scope='CLI local preparation/upload only; excludes assistant image creation')

    def _source(self, entry, job_dir):
        source = (self.workspace / entry['source_file']).resolve()
        if not source.is_relative_to(self.workspace):
            raise PipelineBlocked('source_outside_workspace')
        parts = re.split(r'(?m)^## ' + re.escape(entry['source_topic']) + r'\s*$', source.read_text(encoding='utf-8'))
        if len(parts) != 2:
            raise PipelineBlocked('missing_exact_source_heading')
        numbered = re.findall(r'(?m)^(\d+)\. (.+)$', parts[1].split('\n## ', 1)[0])
        if [n for n, _ in numbered] != ['1', '2', '3', '4']:
            raise PipelineBlocked('four_slide_source_required')
        lines = [line for _, line in numbered]
        if any(line != line.strip() for line in lines):
            raise PipelineBlocked('original_whitespace_requires_review')
        locked = dict(source_file=entry['source_file'], topic=entry['source_topic'], lines=lines,
                      fingerprint=hashlib.sha256(_encoded(lines)).hexdigest())
        _immutable(job_dir / 'source-lock.json', _encoded(locked))
        text = '## ' + entry['source_topic'] + '\n\n' + '\n'.join(f'{i+1}. {line}' for i, line in enumerate(lines)) + '\n'
        _immutable(job_dir / 'locked-draft.md', text.encode('utf-8'))
        return lines

    def _manifest(self, entry, lines, recipe, job_dir):
        if not isinstance(recipe, dict) or recipe.get('source_issues'):
            raise PipelineBlocked('source_correction_requires_user_approval')
        recipes = recipe.get('slides')
        if not isinstance(recipes, list) or len(recipes) != 4:
            raise PipelineBlocked('four_slide_recipe_required')
        slides = []
        for i, (line, item) in enumerate(zip(lines, recipes, strict=True)):
            if (not isinstance(item, dict) or set(item) != {'title', 'fun_fact', 'image_prompt'} or
                    any(not isinstance(v, str) or not v.strip() for v in item.values())):
                raise PipelineBlocked('invalid_separate_recipe_fields')
            slides.append(dict(title=item['title'], explanation=line, narration=line, fun_fact=item['fun_fact'],
                               image=f'images/slide-{i+1:02d}.png'))
        record = {k: entry[k] for k in ('slug', 'question', 'title', 'category', 'sort_order')}
        record.update(summary=entry.get('summary', f"A four-slide private lesson about {entry['title']}"),
                      featured_rank=None, cover='images/slide-01.png', slides=slides)
        target = job_dir / 'manifest.json'
        _immutable(target, _encoded([record]))
        return target

    def _confirmed(self, job, receipt, timings):
        if not valid_receipt(receipt, job['entry']['slug'], job['review_fingerprint']):
            raise UploadUncertain('Stored receipt does not match the prepared private version')
        preview = self.config['server_url'] + receipt['preview_url']
        stages = {k: timings.get(k, 0) for k in ('setup', 'pictures', 'narration', 'packaging', 'upload')}
        slowest = max(stages, key=stages.get)
        updated = self.queue.update(job['job_id'], 'ready', preview_url=preview, timings=timings, slowest_stage=slowest)
        return self._ready(updated)

    def _reconcile(self, job, retry_upload=False):
        started = time.perf_counter()
        receipt = self._client().lookup(job['entry']['slug'])
        if receipt is not None:
            return self._confirmed(job, receipt, {**job.get('timings', {}), 'reconcile': time.perf_counter()-started})
        if not retry_upload:
            raise PipelineBlocked('upload_unconfirmed', 'No matching saved receipt yet; no automatic POST retry')
        raw = Path(job['package_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != job['packet_sha256']:
            raise PipelineBlocked('package_changed_after_upload')
        receipt = self._client().upload(raw, job['entry']['slug'], job['review_fingerprint'])
        return self._confirmed(job, receipt, {**job.get('timings', {}), 'reconcile_and_explicit_retry': time.perf_counter()-started})

    def run_next(self, *, resume=False, retry_upload=False, lookahead=False):
        previous = self.queue.snapshot()['lookahead' if lookahead else 'current']
        result = self._run_slot(resume=resume, retry_upload=retry_upload, lookahead=lookahead)
        if not lookahead and result['status'] == 'ready' and (previous is None or previous['status'] != 'ready'):
            # Spawn only after the completed run released its execution lock.
            # A child startup failure cannot undo the confirmed current review.
            try:
                automatic = self._automatic_lookahead()
                if automatic:
                    result['lookahead'] = automatic
            except Exception:
                result['lookahead'] = dict(status='explicit_resume_required')
        return result

    def _run_slot(self, *, resume=False, retry_upload=False, lookahead=False):
        slot = 'lookahead' if lookahead else 'current'
        snapshot = self.queue.snapshot()
        current = snapshot[slot]
        if lookahead:
            if snapshot['current'] is None or snapshot['current']['status'] != 'ready':
                raise PipelineBlocked('lookahead_requires_current_ready')
            candidate = current['entry'] if current else self.queue.next_lookahead_entry()
            if candidate is not None and candidate['slug'] != self.config.get('lookahead_slug'):
                raise PipelineBlocked('lookahead_requires_exact_configured_slug')
        if current and current['status'] == 'ready':
            return self._ready(current)
        if current and not resume:
            return dict(status=current['status'], job_id=current['job_id'], slug=current['entry']['slug'], resume_required=True)
        reconciling = bool(current and current['status'] in {'upload_pending', 'upload_uncertain', 'uncertain'})
        blockers = self._blockers(needs_media=not reconciling)
        if blockers:
            raise PipelineBlocked('setup_required', '; '.join(blockers))
        with self._execution():
            # Recheck under the process-wide lock; a concurrent --next may have claimed it.
            snapshot = self.queue.snapshot()
            saved = snapshot[slot]
            if lookahead:
                if snapshot['current'] is None or snapshot['current']['status'] != 'ready':
                    raise PipelineBlocked('lookahead_requires_current_ready')
                candidate = saved['entry'] if saved else self.queue.next_lookahead_entry()
                if candidate is not None and candidate['slug'] != self.config.get('lookahead_slug'):
                    raise PipelineBlocked('lookahead_requires_exact_configured_slug')
            if saved and saved['status'] == 'ready':
                return self._ready(saved)
            if saved and not resume:
                return dict(status=saved['status'], job_id=saved['job_id'])
            job = saved or (self.queue.claim_lookahead() if lookahead else self.queue.claim_next())
            if job is None:
                return dict(status='queue_complete', preview_url=None)
            if job['status'] in {'upload_pending', 'upload_uncertain', 'uncertain'}:
                return self._reconcile(job, retry_upload)
            started = time.perf_counter()
            job_dir = (self.work_dir / job['job_id']).resolve()
            if not job_dir.is_relative_to(self.work_dir) or job_dir.is_symlink():
                raise PipelineBlocked('invalid_job_directory')
            job_dir.mkdir(parents=True, exist_ok=True)
            timer = _Timing(job_dir)
            upload_boundary_entered = False
            self.queue.update(job['job_id'], 'preparing', job_dir=str(job_dir))
            try:
                try:
                    existing = timer.call('existing_lookup', self._client().lookup, job['entry']['slug'])
                except Exception:
                    raise PipelineBlocked('existing_lookup_uncertain', 'Existing-slug lookup unconfirmed; no generation or upload') from None
                if existing is not None:
                    raise PipelineBlocked('existing_slug_requires_reconcile', 'Existing private lesson requires explicit reconciliation; no overwrite')
                media = self._job_media(job['entry'])
                lines = timer.call('source_lock', self._source, job['entry'], job_dir)
                recipe = timer.call('recipe', media.prepare_recipe, job['entry'], lines, job_dir)
                manifest = self._manifest(job['entry'], lines, recipe, job_dir)
                timer.values['setup'] = time.perf_counter()-started
                image_paths = [job_dir/'images'/f'slide-{i+1:02d}.png' for i in range(4)]
                with ThreadPoolExecutor(max_workers=5) as workers:
                    pictures = [workers.submit(timer.call, f'image_{i+1}', media.make_image,
                                               recipe['slides'][i]['image_prompt'], image_paths[i], i) for i in range(4)]
                    audio = workers.submit(timer.call, 'narration', media.make_audio, manifest, job_dir)
                    for future in pictures:
                        future.result()
                    audio.result()
                timer.values['pictures'] = max(timer.values[f'image_{i+1}'] for i in range(4))
                check = timer.call('image_checks', media.check_images, image_paths)
                if not check.get('ok'):
                    raise PipelineBlocked('image_anomaly_requires_review')
                self.queue.update(job['job_id'], 'assets_ready', image_checks=check)
                report = timer.call('packaging', lambda: build_package(
                    manifest=manifest, draft=job_dir/'locked-draft.md', topic=job['entry']['source_topic'],
                    audio_dir=job_dir, output_dir=job_dir/'package', ffmpeg=self.config['ffmpeg']))
                job = self.queue.update(job['job_id'], 'upload_pending', package_path=report['package_path'],
                                        packet_sha256=report['packet_sha256'], review_fingerprint=report['review_fingerprint'],
                                        timings=dict(timer.values))
                raw = Path(report['package_path']).read_bytes()
                upload_boundary_entered = True
                receipt = timer.call('upload', self._client().upload, raw, job['entry']['slug'], report['review_fingerprint'])
                timer.values['total'] = time.perf_counter()-started
                return self._confirmed(job, receipt, dict(timer.values))
            except Exception as exc:
                if upload_boundary_entered or isinstance(exc, UploadUncertain):
                    try:
                        self.queue.update(job['job_id'], 'upload_uncertain', timings=dict(timer.values), error_code='upload_unconfirmed')
                    except Exception:
                        # upload_pending was persisted before POST. Never replace
                        # an uncertain saved outcome with a retryable media failure.
                        pass
                    if isinstance(exc, UploadUncertain):
                        raise
                    raise UploadUncertain('Upload or local acknowledgement is unconfirmed; reconcile before any explicit retry') from None
                if isinstance(exc, PipelineBlocked):
                    self.queue.update(job['job_id'], 'blocked', timings=dict(timer.values), error_code=exc.code)
                    raise
                self.queue.update(job['job_id'], 'failed', timings=dict(timer.values), error_type=type(exc).__name__)
                raise PipelineBlocked('stage_failed', 'A stage failed; inspect safe receipts before explicit resume') from None

    def _start_lookahead(self):
        if self.config_path is None:
            raise PipelineBlocked('config_path_required_for_lookahead')
        self.work_dir.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).resolve().parents[1] / 'scripts/make_lesson.py'
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (self.work_dir/'lookahead.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(script), '--config', str(self.config_path), '--next', '--lookahead', '--resume'],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=flags, close_fds=True)
        return process.pid

    def _automatic_lookahead(self):
        """Launch one configured, already supplied next lesson; never create art."""
        candidate = self.queue.next_lookahead_entry()
        if (candidate is None or candidate['slug'] != self.config.get('lookahead_slug') or
                self.config_path is None or not self.config_path.is_file()):
            return None
        folder = Path(self.config.get('preparation_dir') or '.') / candidate['slug']
        recipe_path = Path(candidate.get('recipe_file') or self.config.get('recipe_file') or folder/'recipe.json')
        artwork_path = Path(self.config.get('artwork_manifest') or folder/'artwork.json')
        try:
            if not recipe_path.is_file() or not artwork_path.is_file():
                return None
            if self.config.get('require_review_cache') is True and not recipe_path.with_name('review.json').is_file():
                return None
            artwork = json.loads(artwork_path.read_bytes())
            if (not isinstance(artwork, dict) or not isinstance(artwork.get('slides'), list) or
                    len(artwork['slides']) != 4 or
                    any(not (artwork_path.parent / slide['path']).is_file() for slide in artwork['slides'])):
                return None
        except (OSError, ValueError, TypeError, KeyError):
            return None
        with self._execution():
            # A concurrent command may have claimed the slot while inputs were
            # checked. Its work must never launch a second child.
            if self.queue.next_lookahead_entry() != candidate:
                return None
            job = self.queue.claim_lookahead()
            if job is None:
                return None
        return dict(status='started', job_id=job['job_id'], pid=self._start_lookahead())

    def approve(self, slug, fingerprint, *, lookahead=True):
        approved = self.queue.approve(slug, fingerprint)
        result = dict(approved=True, slug=slug, already_approved=approved['already_approved'], production_approval_changed=False)
        if not approved['changed'] or not lookahead:
            return result
        # Approval promotes an existing lookahead locally. A new automatic
        # child is considered only for a ready promoted current and exact inputs.
        automatic = self._automatic_lookahead()
        return {**result, 'lookahead': automatic} if automatic else result
