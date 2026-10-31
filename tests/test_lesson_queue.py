"""Local queue invariants; no generation, storage connection, or production writes."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import pytest

from lesson_platform.lesson_queue import LessonQueue, QueueError, QueueLockError


CATALOG = Path(__file__).parents[1] / 'assets/editorial/lesson-queue.json'


@pytest.fixture
def queue(tmp_path):
    return LessonQueue(CATALOG, tmp_path / 'durable/state.json')


def test_full_catalog_exact_order_and_source_references(queue):
    entries = queue.snapshot()['entries']
    assert len(entries) == 30
    assert [e['position'] for e in entries] == list(range(1, 31))
    assert [e['status'] for e in entries[:16]] == ['approved'] * 16
    assert [e['slug'] for e in entries[:18]] == [
        'why-is-the-sky-blue', 'how-do-rockets-fly', 'how-does-a-volcano-erupt',
        'how-octopuses-change-color', 'how-the-internet-works', 'how-diwali-is-celebrated',
        'how-does-rain-happen', 'jupiter-the-biggest-planet', 'how-do-fireflies-glow',
        'platypuses-lay-eggs', 'where-jumping-spiders-live', 'how-plants-eat-sunlight',
        'why-some-eggs-have-baby-chicks', 'how-your-digestive-system-works',
        'how-websites-are-built', 'how-tv-shows-are-made', 'what-is-a-dosa', 'what-is-taekwondo']
    assert entries[9]['title'] == 'What Makes a Platypus Special?'
    assert entries[9]['source_topic'] == 'Platypuses Lay Eggs!'
    assert entries[18]['source_topic'] == 'Why Do Owls Say Hoo?'
    assert entries[-1]['source_topic'] == 'The Biggest Countries on Earth'
    assert all(e['source_file'].startswith('Howstuffworks-claudeway/docs/editorial/') for e in entries)


def test_next_is_readonly_and_starts_after_approved_sixteen(queue):
    assert queue.next_entry()['slug'] == 'what-is-a-dosa'
    assert not queue.state_path.exists()
    assert not queue.state_path.parent.exists()
    assert queue.snapshot()['current'] is None
    assert not queue.state_path.exists()


def test_claim_is_one_slot_and_restart_preserves_it(queue):
    first = queue.claim_next()
    assert first['status'] == 'claimed'
    assert first['entry']['position'] == 17
    assert queue.claim_next() == first
    assert queue.next_entry() is None
    reopened = LessonQueue(CATALOG, queue.state_path)
    assert reopened.snapshot()['current'] == first
    assert reopened.claim_next() == first


def test_concurrent_claims_never_allocate_two_jobs(queue):
    def claim(_):
        try:
            return LessonQueue(CATALOG, queue.state_path).claim_next()
        except QueueLockError:
            return None
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(claim, range(36)))
    jobs = [r for r in results if r is not None]
    assert jobs
    assert len({r['job_id'] for r in jobs}) == 1
    assert len(queue.snapshot()['jobs']) == 1


def test_ready_requires_fingerprint_and_exact_version_approval(queue):
    job = queue.claim_next()
    with pytest.raises(QueueError, match='fingerprint'):
        queue.update(job['job_id'], 'ready', review_url='https://example.test/review/dosa')
    queue.update(job['job_id'], 'ready', review_fingerprint='version-one', review_url='https://example.test/review/dosa')
    with pytest.raises(QueueError, match='fingerprint'):
        queue.approve(job['entry']['slug'], 'old-version')
    current = queue.snapshot()['current']
    assert current['review_url'] == 'https://example.test/review/dosa'
    assert queue.claim_next() == current
    approved = queue.approve(job['entry']['slug'], 'version-one')
    assert approved['status'] == 'approved'
    assert approved['already_approved'] is False
    assert queue.snapshot()['current'] is None
    assert queue.next_entry()['slug'] == 'what-is-taekwondo'


def test_duplicate_approval_does_not_clear_next_job_or_trigger_lookahead(queue):
    first = queue.claim_next()
    queue.update(first['job_id'], 'ready', review_fingerprint='one')
    queue.approve(first['entry']['slug'], 'one')
    second = queue.claim_next()
    duplicate = queue.approve(first['entry']['slug'], 'one')
    assert duplicate['already_approved'] is True
    assert queue.snapshot()['current'] == second
    assert len(queue.snapshot()['jobs']) == 2
    with pytest.raises(QueueError):
        queue.approve(first['entry']['slug'], 'different')


@pytest.mark.parametrize('status', ['blocked', 'uncertain', 'failed'])
def test_failed_jobs_need_explicit_resume_and_keep_fields(queue, status):
    first = queue.claim_next()
    saved = queue.update(first['job_id'], status, error='Keep receipt', provider_marker='uncertain-1')
    restarted = LessonQueue(CATALOG, queue.state_path)
    assert restarted.claim_next() == saved
    assert restarted.next_entry() is None
    with pytest.raises(QueueError):
        restarted.approve(first['entry']['slug'], 'one')
    resumed = restarted.update(first['job_id'], 'claimed', resumed=True)
    assert resumed['job_id'] == first['job_id']
    assert resumed['provider_marker'] == 'uncertain-1'


def test_held_cube_preserves_sixteen_step_publication_gate(queue):
    entries = queue.snapshot()['entries']
    cube = next(e for e in entries if e['status'] == 'held')
    assert cube['source_topic'] == 'Solve a 2×2 Rubik’s Cube — move-by-move draft'
    assert cube['source_step_count'] == 16
    assert 'physical' in cube['held_reason']
    produced = []
    while queue.next_entry():
        job = queue.claim_next()
        produced.append(job['entry']['slug'])
        queue.update(job['job_id'], 'ready', review_fingerprint=job['job_id'])
        queue.approve(job['entry']['slug'], job['job_id'])
    assert len(produced) == 13
    assert cube['slug'] not in produced
    assert queue.claim_next() is None


def test_stale_lock_is_never_deleted_or_reclaimed(queue):
    queue.state_path.parent.mkdir(parents=True)
    lock = queue.state_path.with_suffix('.json.lock')
    lock.write_text('{"token":"dead-owner","pid":999999999}', encoding='utf-8')
    with pytest.raises(QueueLockError):
        queue.claim_next()
    assert json.loads(lock.read_text())['token'] == 'dead-owner'
    assert not queue.state_path.exists()


@pytest.mark.parametrize('mutation', [
    lambda entries: entries[1].update(slug=entries[0]['slug']),
    lambda entries: entries[1].update(position=1),
    lambda entries: entries[0].update(source_file='../outside.md'),
    lambda entries: entries[0].update(source_file='C:/outside.md'),
    lambda entries: entries[0].update(source_file='Howstuffworks-claudeway/docs/editorial/not-approved.md'),
    lambda entries: entries[0].update(source_topic='Missing heading'),
    lambda entries: entries[0].update(status='ready'),
    lambda entries: entries[0].update(sort_order=True),
    lambda entries: next(e for e in entries if e['status'] == 'held').update(held_reason=''),
    lambda entries: next(e for e in entries if e['status'] == 'held').update(source_step_count=4),
])
def test_invalid_catalog_fails_closed(tmp_path, mutation):
    entries = copy.deepcopy(json.loads(CATALOG.read_text(encoding='utf-8')))
    mutation(entries)
    path = tmp_path / 'bad-catalog.json'
    path.write_text(json.dumps(entries), encoding='utf-8')
    with pytest.raises(QueueError):
        LessonQueue(path, tmp_path / 'state.json')


@pytest.mark.parametrize('content', ['{', 'null', '[]', '{}', '{"version":1,"version":2}'])
def test_malformed_state_is_preserved(queue, content):
    queue.state_path.parent.mkdir(parents=True)
    queue.state_path.write_text(content, encoding='utf-8')
    with pytest.raises(QueueError):
        queue.claim_next()
    assert queue.state_path.read_text(encoding='utf-8') == content


def test_caller_cannot_mutate_snapshot_or_job_identity(queue):
    job = queue.claim_next()
    job['entry']['slug'] = 'changed'
    assert queue.snapshot()['current']['entry']['slug'] == 'what-is-a-dosa'
    with pytest.raises(QueueError):
        queue.update(job['job_id'], 'claimed', entry={})
    with pytest.raises(QueueError):
        queue.update(job['job_id'], 'approved')


def test_state_catalog_change_and_multiple_current_jobs_fail_closed(queue, tmp_path):
    queue.claim_next()
    state = json.loads(queue.state_path.read_text())
    clone = copy.deepcopy(next(iter(state['jobs'].values())))
    clone['job_id'] = 'second-job'
    state['jobs']['second-job'] = clone
    queue.state_path.write_text(json.dumps(state), encoding='utf-8')
    with pytest.raises(QueueError):
        queue.snapshot()


def test_changed_ready_fingerprint_invalidates_earlier_review(queue):
    job = queue.claim_next()
    queue.update(job['job_id'], 'ready', review_fingerprint='first')
    queue.update(job['job_id'], 'ready', review_fingerprint='replacement')
    with pytest.raises(QueueError, match='fingerprint'):
        queue.approve(job['entry']['slug'], 'first')
    assert queue.approve(job['entry']['slug'], 'replacement')['changed'] is True
    assert queue.approve(job['entry']['slug'], 'replacement')['changed'] is False


@pytest.mark.parametrize('status', ['preparing', 'assets_ready', 'upload_pending', 'upload_uncertain'])
def test_orchestrator_stages_all_retain_the_one_slot(queue, status):
    job = queue.claim_next()
    updated = queue.update(job['job_id'], status, job_dir='tmp/lesson-pipeline/jobs/one',
                           timings={'recipe': 1.2}, preview_url='https://example.test/private')
    assert queue.next_entry() is None
    assert queue.claim_next() == updated


def test_failed_atomic_replace_preserves_previous_state_and_releases_owned_lock(queue, monkeypatch):
    import lesson_platform.lesson_queue as module
    job = queue.claim_next()
    original = queue.state_path.read_bytes()
    def failed_replace(*args):
        raise OSError('simulated disk replacement failure')
    monkeypatch.setattr(module.os, 'replace', failed_replace)
    with pytest.raises(OSError, match='simulated'):
        queue.update(job['job_id'], 'blocked', error='saved only on success')
    assert queue.state_path.read_bytes() == original
    assert not queue.lock_path.exists()
    assert not list(queue.state_path.parent.glob('*.tmp'))


def test_changed_catalog_requires_explicit_recovery(queue, tmp_path):
    queue.claim_next()
    entries = json.loads(CATALOG.read_text(encoding='utf-8'))
    entries[16]['category'] = 'Changed metadata'
    changed = tmp_path / 'changed.json'
    changed.write_text(json.dumps(entries), encoding='utf-8')
    with pytest.raises(QueueError, match='catalog changed'):
        LessonQueue(changed, queue.state_path).snapshot()


def test_small_fixture_catalog_supported(tmp_path):
    entries = json.loads(CATALOG.read_text(encoding='utf-8'))[16:18]
    for position, entry in enumerate(entries, 1):
        entry['position'] = position
    catalog = tmp_path / 'fixture.json'
    catalog.write_text(json.dumps(entries), encoding='utf-8')
    queue = LessonQueue(catalog, tmp_path / 'not-created/state.json')
    assert queue.next_entry()['slug'] == 'what-is-a-dosa'
    assert not queue.state_path.parent.exists()


def test_one_lookahead_survives_restart_and_promotes_only_after_exact_approval(queue):
    current = queue.claim_next()
    with pytest.raises(QueueError):
        queue.claim_lookahead()
    queue.update(current['job_id'], 'ready', review_fingerprint='current-version')
    current = queue.snapshot()['current']
    ahead = queue.claim_lookahead()
    assert ahead['entry']['position'] == 18
    queue.update(ahead['job_id'], 'upload_uncertain', marker='preserve-me')
    reopened = LessonQueue(CATALOG, queue.state_path)
    assert reopened.snapshot()['current'] == current
    assert reopened.claim_lookahead()['job_id'] == ahead['job_id']
    assert len(reopened.snapshot()['jobs']) == 2
    with pytest.raises(QueueError):
        reopened.approve(ahead['entry']['slug'], 'anything')
    with pytest.raises(QueueError):
        reopened.approve(current['entry']['slug'], 'old-version')
    reopened.approve(current['entry']['slug'], 'current-version')
    state = reopened.snapshot()
    assert state['current']['job_id'] == ahead['job_id']
    assert state['current']['status'] == 'upload_uncertain'
    assert state['current']['marker'] == 'preserve-me'
    assert state['lookahead'] is None


def test_concurrent_lookahead_claims_never_allocate_third_job(queue):
    current = queue.claim_next()
    queue.update(current['job_id'], 'ready', review_fingerprint='reviewed-version')
    def claim(_):
        try:
            return LessonQueue(CATALOG, queue.state_path).claim_lookahead()
        except QueueLockError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [job for job in pool.map(claim, range(24)) if job]
    assert len({job['job_id'] for job in jobs}) == 1
    assert len(queue.snapshot()['jobs']) == 2


def test_claim_selected_uses_full_catalog_and_isolated_state_without_changing_main(queue, tmp_path):
    current = queue.claim_next()
    queue.update(current['job_id'], 'ready', review_fingerprint='main-review')
    queue.claim_lookahead()
    main_bytes, catalog_bytes = queue.state_path.read_bytes(), CATALOG.read_bytes()
    selected = LessonQueue(CATALOG, tmp_path/'selected/state.json')
    # Select the final pending topic without skipping/approving preceding drafts.
    slug = selected.snapshot()['entries'][-1]['slug']
    job = selected.claim_selected(slug)
    assert job['entry']['position'] == 30 and job['status'] == 'claimed'
    assert selected.snapshot()['current'] == job
    assert selected.snapshot()['lookahead'] is None
    assert len(selected.snapshot()['jobs']) == 1
    assert len(selected.snapshot()['entries']) == 30
    assert queue.state_path.read_bytes() == main_bytes
    assert CATALOG.read_bytes() == catalog_bytes


def test_selected_claim_is_idempotent_after_restart_and_preserves_uncertainty(queue):
    job = queue.claim_selected('what-is-taekwondo')
    saved = queue.update(job['job_id'], 'upload_uncertain', error_code='unconfirmed', marker='keep')
    before = queue.state_path.read_bytes()
    reopened = LessonQueue(CATALOG, queue.state_path)
    assert reopened.claim_selected('what-is-taekwondo') == saved
    assert queue.state_path.read_bytes() == before
    assert reopened.claim_next() == saved
    with pytest.raises(QueueError): reopened.claim_selected('what-is-a-dosa')
    assert queue.state_path.read_bytes() == before


@pytest.mark.parametrize('slug', ['unknown-lesson', 'why-is-the-sky-blue', '', None, [], True])
def test_selected_claim_rejects_invalid_unknown_or_catalog_approved_without_writing(queue, slug):
    with pytest.raises(QueueError): queue.claim_selected(slug)
    assert not queue.state_path.exists()


def test_selected_claim_rejects_held_and_locally_approved_jobs(queue):
    held = next(entry['slug'] for entry in queue.snapshot()['entries'] if entry['status'] == 'held')
    with pytest.raises(QueueError): queue.claim_selected(held)
    job = queue.claim_selected('what-is-taekwondo')
    queue.update(job['job_id'], 'ready', review_fingerprint='selected-review')
    queue.approve('what-is-taekwondo', 'selected-review')
    before = queue.state_path.read_bytes()
    with pytest.raises(QueueError): queue.claim_selected('what-is-taekwondo')
    with pytest.raises(QueueError): queue.claim_selected('what-is-a-dosa')
    assert queue.state_path.read_bytes() == before


def test_selected_claim_rejects_queue_with_lookahead_without_mutation(queue):
    current = queue.claim_next()
    queue.update(current['job_id'], 'ready', review_fingerprint='main-review')
    queue.claim_lookahead()
    before = queue.state_path.read_bytes()
    with pytest.raises(QueueError): queue.claim_selected(current['entry']['slug'])
    assert queue.state_path.read_bytes() == before


def test_concurrent_selected_claims_do_not_allocate_different_jobs(queue):
    def claim(index):
        try:
            return LessonQueue(CATALOG, queue.state_path).claim_selected(
                'what-is-taekwondo' if index % 2 else 'what-is-a-dosa')
        except QueueError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [job for job in pool.map(claim, range(24)) if job]
    assert jobs and len({job['job_id'] for job in jobs}) == 1
    assert len(queue.snapshot()['jobs']) == 1


@pytest.mark.parametrize('mutation', [
    lambda state: state.update(lookahead_job_id=state['current_job_id']),
    lambda state: state.update(lookahead_job_id='missing'),
    lambda state: state.update(current_job_id=None),
    lambda state: state.update(lookahead_job_id=[]),
])
def test_invalid_lookahead_slot_state_fails_closed(queue, mutation):
    current = queue.claim_next()
    queue.update(current['job_id'], 'ready', review_fingerprint='reviewed-version')
    queue.claim_lookahead()
    state = json.loads(queue.state_path.read_text())
    mutation(state)
    queue.state_path.write_text(json.dumps(state))
    with pytest.raises(QueueError):
        queue.snapshot()


@pytest.mark.parametrize('content', ['null', '{}', '[null]', '[{"slug":"incomplete"}]', '{',
                                     '[{"position":1,"position":2}]'])
def test_malformed_catalogs_fail_closed(tmp_path, content):
    catalog = tmp_path / 'malformed.json'
    catalog.write_text(content, encoding='utf-8')
    with pytest.raises(QueueError):
        LessonQueue(catalog, tmp_path / 'state.json')


@pytest.mark.parametrize('mutation', [
    lambda job: job.update(status=[]),
    lambda job: job['entry'].update(slug=[]),
    lambda job: job.update(review_fingerprint=None, status='ready'),
    lambda job: job.update(status='approved'),
])
def test_invalid_saved_job_shapes_fail_closed(queue, mutation):
    job = queue.claim_next()
    state = json.loads(queue.state_path.read_text())
    mutation(state['jobs'][job['job_id']])
    content = json.dumps(state)
    queue.state_path.write_text(content, encoding='utf-8')
    with pytest.raises(QueueError):
        queue.snapshot()
    assert queue.state_path.read_text(encoding='utf-8') == content

