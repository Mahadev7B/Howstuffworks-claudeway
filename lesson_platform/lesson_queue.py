"""Durable, local-only, one-lesson review queue.

Catalog approval records already reviewed lessons. New approvals apply only to a
specific saved job fingerprint and never touch database or publication flags.
An existing job (including failure/uncertainty) occupies the slot until approved;
resumption is an explicit update of that same job, never a new automatic claim.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import uuid


class QueueError(ValueError):
    """Invalid catalog, durable state, or requested state transition."""


class QueueLockError(QueueError):
    """Another owner holds the lock; inspection/manual recovery is required."""


_ACTIVE = frozenset({'claimed', 'preparing', 'assets_ready', 'upload_pending',
                     'upload_uncertain', 'ready', 'blocked', 'uncertain', 'failed'})
_SOURCES = frozenset({
    'Howstuffworks-claudeway/docs/editorial/first-15-lesson-script-revisions.md',
    'Howstuffworks-claudeway/docs/editorial/remaining-15-lesson-script-revisions.md',
})
_WORKSPACE = Path(__file__).resolve().parents[2]
_CUBE = 'Solve a 2×2 Rubik’s Cube — move-by-move draft'


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QueueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(QueueError('Nonfinite JSON number')))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueueError(f'Cannot read valid JSON: {path}') from exc


def _encoded(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise QueueError('Queue values must be finite JSON data') from exc


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class LessonQueue:
    def __init__(self, catalog_path, state_path):
        self.catalog_path = Path(catalog_path).resolve()
        self.state_path = Path(state_path).resolve()
        if self.catalog_path == self.state_path:
            raise QueueError('Catalog and mutable state must be separate files')
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + '.lock')
        self._catalog = self._load_catalog()
        self._by_slug = {entry['slug']: entry for entry in self._catalog}
        self._fingerprint = hashlib.sha256(_encoded(self._catalog)).hexdigest()

    def _load_catalog(self):
        entries = _read_json(self.catalog_path)
        if not isinstance(entries, list) or not entries:
            raise QueueError('Catalog must be a nonempty list')
        slugs, positions, refs, orders = set(), set(), set(), set()
        headings = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise QueueError('Catalog entry must be an object')
            for key in ('slug', 'title', 'question', 'category', 'source_file', 'source_topic'):
                if not _text(entry.get(key)):
                    raise QueueError(f'Catalog entry requires {key}')
            if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', entry['slug']):
                raise QueueError('Invalid catalog slug')
            if not _int(entry.get('position')) or not _int(entry.get('sort_order')):
                raise QueueError('Catalog positions and sort_order must be positive integers')
            if not isinstance(entry.get('status'), str) or entry['status'] not in {'approved', 'pending', 'held'}:
                raise QueueError('Catalog status must be approved, pending, or held')
            if entry['status'] == 'held' and not _text(entry.get('held_reason')):
                raise QueueError('Held entry requires a held_reason')
            ref = (entry['source_file'], entry['source_topic'])
            if entry['slug'] in slugs or entry['position'] in positions or ref in refs or entry['sort_order'] in orders:
                raise QueueError('Duplicate catalog slug, position, sort order, or source reference')
            slugs.add(entry['slug'])
            positions.add(entry['position'])
            orders.add(entry['sort_order'])
            refs.add(ref)
            source = PurePosixPath(entry['source_file'])
            if (source.is_absolute() or '..' in source.parts or '\\' in entry['source_file']
                    or ':' in entry['source_file'] or entry['source_file'] not in _SOURCES):
                raise QueueError('Source reference must name an approved workspace editorial source')
            path = (_WORKSPACE / source).resolve()
            if not path.is_relative_to(_WORKSPACE):
                raise QueueError('Source reference escapes workspace')
            if path not in headings:
                try:
                    text = path.read_text(encoding='utf-8')
                except (OSError, UnicodeError) as exc:
                    raise QueueError(f'Editorial source is unavailable: {source}') from exc
                headings[path] = re.findall(r'^## (.+)$', text, re.MULTILINE)
            if headings[path].count(entry['source_topic']) != 1:
                raise QueueError('Source topic must match one exact editorial heading')
            if entry['source_topic'] == _CUBE:
                if entry['status'] != 'held' or entry.get('source_step_count') != 16:
                    raise QueueError('Cube must retain the 16-step draft and publication hold')
        if positions != set(range(1, len(entries) + 1)):
            raise QueueError('Catalog positions must be contiguous starting at one')
        return sorted(entries, key=lambda entry: entry['position'])

    def _empty_state(self):
        return dict(version=1, catalog_fingerprint=self._fingerprint, current_job_id=None, jobs={})

    def _read_state(self):
        if not self.state_path.exists():
            return self._empty_state()
        state = _read_json(self.state_path)
        self._validate_state(state)
        return state

    def _validate_state(self, state):
        if (not isinstance(state, dict) or type(state.get('version')) is not int or state['version'] != 1
                or state.get('catalog_fingerprint') != self._fingerprint
                or 'current_job_id' not in state or not isinstance(state.get('jobs'), dict)):
            raise QueueError('Malformed state or catalog changed; explicit recovery required')
        current, lookahead = state['current_job_id'], state.get('lookahead_job_id')
        if any(value is not None and not _text(value) for value in (current, lookahead)):
            raise QueueError('Invalid current or lookahead job ID')
        if lookahead is not None and (current is None or current == lookahead):
            raise QueueError('Lookahead requires a distinct current job')
        active, seen = [], set()
        for job_id, job in state['jobs'].items():
            if not _text(job_id) or not isinstance(job, dict) or job.get('job_id') != job_id:
                raise QueueError('Invalid saved job identity')
            entry = job.get('entry')
            if not isinstance(entry, dict) or not _text(entry.get('slug')) or entry['slug'] not in self._by_slug:
                raise QueueError('Invalid saved job entry')
            slug = entry['slug']
            if (slug in seen or entry != self._by_slug[slug] or entry['status'] != 'pending'
                    or not isinstance(job.get('status'), str) or job['status'] not in _ACTIVE | {'approved'}):
                raise QueueError('Invalid saved job status, duplicate entry, or catalog mismatch')
            seen.add(slug)
            if not _text(job.get('created_at')) or not _text(job.get('updated_at')):
                raise QueueError('Saved job timestamps are required')
            if job['status'] in {'ready', 'approved'} and not _text(job.get('review_fingerprint')):
                raise QueueError('Ready or approved job requires review_fingerprint')
            if job['status'] == 'approved' and not _text(job.get('approved_at')):
                raise QueueError('Approved job requires approved_at')
            if job['status'] in _ACTIVE:
                active.append(job_id)
        slots = {value for value in (current, lookahead) if value is not None}
        if set(active) != slots or len(active) > 2:
            raise QueueError('Only current and one lookahead job are allowed; explicit recovery required')

    @contextmanager
    def _locked(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        try:
            fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise QueueLockError(f'Queue lock exists: {self.lock_path}; never reclaim automatically') from exc
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(_encoded(dict(token=token, pid=os.getpid(), created_at=_now())))
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            # Only the creating owner may release this lock. Old locks are not
            # considered safe merely because a PID is absent or a clock expired.
            lock = _read_json(self.lock_path)
            if not isinstance(lock, dict) or lock.get('token') != token:
                raise QueueLockError('Queue lock ownership changed; lock preserved')
            self.lock_path.unlink()

    def _write_state(self, state):
        self._validate_state(state)
        data = _encoded(state)
        fd, temporary = tempfile.mkstemp(prefix=self.state_path.name + '.', suffix='.tmp', dir=self.state_path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            # Flush the containing directory where supported (POSIX). Windows
            # file data was flushed above and replacement is on the same volume.
            if os.name != 'nt':
                directory = os.open(self.state_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _next(self, state, lookahead=False):
        if not lookahead and state['current_job_id'] is not None:
            return None
        completed = {job['entry']['slug'] for job in state['jobs'].values()}
        return next((entry for entry in self._catalog
                     if entry['status'] == 'pending' and entry['slug'] not in completed), None)

    def next_entry(self):
        """Read-only next candidate, or None when the review slot is occupied."""
        return deepcopy(self._next(self._read_state()))

    def next_lookahead_entry(self):
        """Read-only candidate only while a current ready review occupies its slot."""
        state = self._read_state()
        current = state['jobs'].get(state['current_job_id'])
        if current is None or current['status'] != 'ready' or state.get('lookahead_job_id') is not None:
            return None
        return deepcopy(self._next(state, lookahead=True))

    def claim_lookahead(self):
        """Atomically reserve at most one next job without changing current review."""
        with self._locked():
            state = self._read_state()
            if state.get('lookahead_job_id') is not None:
                return deepcopy(state['jobs'][state['lookahead_job_id']])
            current = state['jobs'].get(state['current_job_id'])
            if current is None or current['status'] != 'ready':
                raise QueueError('Lookahead requires the current ready job')
            entry = self._next(state, lookahead=True)
            if entry is None:
                return None
            job_id, timestamp = uuid.uuid4().hex, _now()
            job = dict(job_id=job_id, entry=deepcopy(entry), status='claimed',
                       created_at=timestamp, updated_at=timestamp)
            state['jobs'][job_id] = job
            state['lookahead_job_id'] = job_id
            self._write_state(state)
            return deepcopy(job)

    def claim_next(self):
        """Atomically claim one candidate or return the existing saved job."""
        with self._locked():
            state = self._read_state()
            if state['current_job_id'] is not None:
                return deepcopy(state['jobs'][state['current_job_id']])
            entry = self._next(state)
            if entry is None:
                return None
            job_id = uuid.uuid4().hex
            timestamp = _now()
            job = dict(job_id=job_id, entry=deepcopy(entry), status='claimed',
                       created_at=timestamp, updated_at=timestamp)
            state['jobs'][job_id] = job
            state['current_job_id'] = job_id
            self._write_state(state)
            return deepcopy(job)

    def claim_selected(self, slug):
        """Reserve one explicit pending slug in an isolated, otherwise empty state.

        Callers supply a separate state/work directory for each batch selection;
        this method does not advance, approve, or rewrite other queue states.
        Repeating the same selection returns its saved job, including uncertainty.
        """
        if not _text(slug) or slug not in self._by_slug or self._by_slug[slug]['status'] != 'pending':
            raise QueueError('Selected slug must name a pending, non-held catalog lesson')
        with self._locked():
            state = self._read_state()
            current = state['jobs'].get(state['current_job_id'])
            if (current is not None and current['entry']['slug'] == slug and
                    len(state['jobs']) == 1 and state.get('lookahead_job_id') is None):
                return deepcopy(current)
            if state['jobs'] or state['current_job_id'] is not None or state.get('lookahead_job_id') is not None:
                raise QueueError('Selected claim requires empty isolated state or its identical current selection')
            job_id, timestamp = uuid.uuid4().hex, _now()
            job = dict(job_id=job_id, entry=deepcopy(self._by_slug[slug]), status='claimed',
                       created_at=timestamp, updated_at=timestamp)
            state['jobs'][job_id] = job
            state['current_job_id'] = job_id
            self._write_state(state)
            return deepcopy(job)

    def update(self, job_id, status, **fields):
        """Update the same current job; a claimed update is explicit resumption."""
        if not isinstance(status, str) or status not in _ACTIVE:
            raise QueueError('Unknown active job status; use approve for approval')
        if set(fields) & {'job_id', 'entry', 'status', 'created_at', 'updated_at', 'approved_at'}:
            raise QueueError('Cannot replace job identity or reserved status fields')
        _encoded(fields)
        with self._locked():
            state = self._read_state()
            if not isinstance(job_id, str) or job_id not in (state['current_job_id'], state.get('lookahead_job_id')):
                raise QueueError('Only the current or lookahead saved job may be updated')
            job = state['jobs'][job_id]
            job.update(deepcopy(fields), status=status, updated_at=_now())
            self._write_state(state)
            return deepcopy(job)

    def approve(self, slug, fingerprint):
        """Approve the exact ready version locally; duplicates never clear a newer job.

        The return value includes already_approved. The orchestrator must only
        consider lookahead when this is False. This method never starts work.
        """
        if not _text(slug) or not _text(fingerprint):
            raise QueueError('Approval requires a slug and exact review fingerprint')
        with self._locked():
            state = self._read_state()
            for previous in state['jobs'].values():
                if previous['entry']['slug'] == slug and previous['status'] == 'approved':
                    if previous['review_fingerprint'] != fingerprint:
                        raise QueueError('Approval fingerprint does not match approved version')
                    return dict(deepcopy(previous), already_approved=True, changed=False)
            job = state['jobs'].get(state['current_job_id'])
            if job is None or job['entry']['slug'] != slug or job['status'] != 'ready':
                raise QueueError('Approval requires the current ready job')
            if job['review_fingerprint'] != fingerprint:
                raise QueueError('Approval fingerprint does not match current ready version')
            timestamp = _now()
            job.update(status='approved', approved_at=timestamp, updated_at=timestamp)
            state['current_job_id'] = state.get('lookahead_job_id')
            if 'lookahead_job_id' in state:
                state['lookahead_job_id'] = None
            self._write_state(state)
            return dict(deepcopy(job), already_approved=False, changed=True)

    def snapshot(self):
        """Read-only durable state plus current job and effective catalog statuses."""
        state = self._read_state()
        result = deepcopy(state)
        result['current'] = deepcopy(state['jobs'].get(state['current_job_id']))
        result['lookahead'] = deepcopy(state['jobs'].get(state.get('lookahead_job_id')))
        result['entries'] = deepcopy(self._catalog)
        status_by_slug = {job['entry']['slug']: job['status'] for job in state['jobs'].values()}
        for entry in result['entries']:
            entry['status'] = status_by_slug.get(entry['slug'], entry['status'])
        return result
