"""Orchestration contracts use local fixtures only: no providers, credentials or HTTP."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import threading

import pytest
from test_private_package import inputs


@pytest.fixture
def pipeline_module():
    assert importlib.util.find_spec('lesson_platform.make_lesson'), 'one-command pipeline missing'
    return importlib.import_module('lesson_platform.make_lesson')


@pytest.fixture
def setup(inputs, monkeypatch):
    args, _ = inputs
    root = args['draft'].parent
    source = root / 'Howstuffworks-claudeway/docs/editorial/first-15-lesson-script-revisions.md'
    source.parent.mkdir(parents=True)
    source.write_text(args['draft'].read_text() + '\n\n## Next topic\n\n1. Next one.\n2. Next two.\n3. Next three.\n4. Next four.\n')
    catalog = []
    for i, (slug, title) in enumerate((('local-fixture', 'Fixture topic'), ('next-fixture', 'Next topic'))):
        catalog.append(dict(position=i+1, sort_order=i+17, slug=slug, title=title, question='A local question?',
                            category='Science', status='pending', source_file=str(source.relative_to(root)).replace('\\', '/'), source_topic=title))
    catalog_path = root / 'catalog.json'
    catalog_path.write_text(json.dumps(catalog))
    queue_module = importlib.import_module('lesson_platform.lesson_queue')
    monkeypatch.setattr(queue_module, '_WORKSPACE', root)
    monkeypatch.setenv('ASKLILOWL_IMPORT_TOKEN', 'fixture-token-never-transmitted-' + 'x'*32)
    config = dict(workspace_root=str(root), catalog=str(catalog_path), state_path=str(root/'state.json'),
                  work_dir=str(root/'jobs'), ffmpeg=str(args['ffmpeg']), server_url='https://www.asklilowl.com',
                  upload_authorized=True)
    barrier = threading.Barrier(5, timeout=4)

    class Media:
        def preflight(self): return {'ok': True, 'errors': [], 'estimated_cost_usd': 0.032}
        def prepare_recipe(self, entry, lines, job_dir):
            return dict(slides=[dict(title=f'Step {i+1}', fun_fact='Separate fact.', image_prompt=f'Picture {i+1}') for i in range(4)], source_issues=[])
        def make_audio(self, manifest, job_dir):
            barrier.wait()
            shutil.copytree(args['audio_dir'], job_dir, dirs_exist_ok=True)
            return json.loads((job_dir/'receipt.json').read_text())
        def make_image(self, prompt, target, index):
            barrier.wait()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/'picture.png', target)
            return {'path': str(target), 'flags': []}
        def check_images(self, paths): return {'ok': True, 'flags': [], 'limitations': ['Fixture; not a semantic review']}

    class Client:
        posts = 0
        def __init__(self, base_url, token): pass
        def upload(self, raw, slug, expected_fingerprint):
            Client.posts += 1
            packet = json.loads(raw)
            assert packet['entry']['lesson']['slides'][0]['narration'] == 'First original line.'
            return dict(slug=slug, fingerprint=expected_fingerprint, slides=4, audio_clips=4,
                        active=False, approved_for_publish=False,
                        preview_url=f'/admin/published-lessons/{slug}/preview')
        def lookup(self, slug): return None
    return config, Media(), Client, root


def test_parallel_assets_single_upload_and_ready_link_reused(pipeline_module, setup):
    config, media, client, root = setup
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    result = runner.run_next()
    assert result['status'] == 'ready'
    assert result['preview_url'] == 'https://www.asklilowl.com/admin/published-lessons/local-fixture/preview'
    assert client.posts == 1
    assert set(('setup', 'pictures', 'narration', 'packaging', 'upload', 'total')) <= result['timings'].keys()
    assert (Path(result['job_dir'])/'timing.jsonl').is_file()
    assert runner.run_next()['preview_url'] == result['preview_url']
    assert client.posts == 1
    assert len(runner.queue.snapshot()['jobs']) == 1
    assert runner.queue.next_entry() is None  # occupied review slot, not another draft


@pytest.mark.parametrize('outcome', ['existing', 'unknown'])
def test_duplicate_or_uncertain_prelookup_blocks_before_recipe_or_media(pipeline_module, setup, outcome):
    config, media, client, root = setup
    def forbidden(*args):
        pytest.fail('Duplicate/unknown slug must stop before recipe and media work')
    media.prepare_recipe = forbidden
    def lookup(self, slug):
        if outcome == 'unknown': raise OSError('Network unavailable')
        return {'slug': slug}
    client.lookup = lookup
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    with pytest.raises(pipeline_module.PipelineBlocked) as blocked:
        runner.run_next()
    assert blocked.value.code == ('existing_slug_requires_reconcile' if outcome == 'existing' else 'existing_lookup_uncertain')
    assert client.posts == 0
    assert runner.queue.snapshot()['current']['status'] == 'blocked'


def test_explicit_lookahead_keeps_current_review_and_never_claims_third(pipeline_module, setup):
    config, media, client, root = setup
    config['lookahead_slug'] = 'next-fixture'
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    ready = runner.run_next()
    current = runner.queue.snapshot()['current']
    # Package fixture audio is specifically bound to the first lesson. Stop at
    # the next recipe boundary to exercise durable slot isolation without fake audio.
    def stop_recipe(*args):
        raise pipeline_module.PipelineBlocked('fixture_recipe_boundary')
    media.prepare_recipe = stop_recipe
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next(lookahead=True)
    state = runner.queue.snapshot()
    assert state['current'] == current
    assert state['lookahead']['entry']['slug'] == 'next-fixture'
    assert state['lookahead']['status'] == 'blocked'
    assert runner.run_next()['preview_url'] == ready['preview_url']
    assert runner.run_next(lookahead=True)['resume_required'] is True
    assert len(runner.queue.snapshot()['jobs']) == 2 and client.posts == 1


def test_lookahead_requires_current_ready_and_exact_configured_slug(pipeline_module, setup):
    config, media, client, root = setup
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next(lookahead=True)
    assert not runner.queue.snapshot()['jobs']
    runner.run_next()
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next(lookahead=True)
    assert len(runner.queue.snapshot()['jobs']) == 1


def test_ready_current_does_not_launch_unprepared_automatic_lookahead(pipeline_module, setup, monkeypatch):
    config, media, client, root = setup
    config['lookahead_slug'] = 'next-fixture'
    runner = pipeline_module.LessonPipeline(config, config_path=root/'config.json', media=media, client_factory=client)
    launches = []
    monkeypatch.setattr(runner, '_start_lookahead', lambda: launches.append('one'))
    assert runner.run_next()['status'] == 'ready'
    assert launches == []


def test_plan_and_disabled_setup_never_claim_or_create_assets(pipeline_module, setup, monkeypatch):
    config, media, client, root = setup
    config.update(upload_authorized=False)
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    plan = runner.plan()
    assert plan['next']['slug'] == 'local-fixture'
    assert plan['blockers']
    assert not (root/'state.json').exists() and not (root/'jobs').exists()
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next()
    assert not (root/'state.json').exists() and client.posts == 0


def test_flagged_original_line_blocks_before_generation(pipeline_module, setup):
    config, media, client, _ = setup
    media.prepare_recipe = lambda *args: {'slides': [], 'source_issues': [{'slide': 1, 'proposed': 'Do not autoapply'}]}
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next()
    assert runner.queue.snapshot()['current']['status'] == 'blocked'
    assert client.posts == 0


def test_unknown_upload_outcome_is_not_posted_again_on_resume(pipeline_module, setup):
    config, media, client, _ = setup
    from lesson_platform.private_http import UploadUncertain
    def uncertain(*args, **kwargs):
        client.posts += 1
        raise UploadUncertain('Fixture unconfirmed')
    client.upload = uncertain
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    with pytest.raises(UploadUncertain): runner.run_next()
    assert runner.queue.snapshot()['current']['status'] == 'upload_uncertain'
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next(resume=True)
    assert client.posts == 1


def test_local_approval_is_version_bound_and_unprepared_next_does_not_spawn(pipeline_module, setup, monkeypatch):
    config, media, client, root = setup
    runner = pipeline_module.LessonPipeline(config, config_path=root/'config.json', media=media, client_factory=client)
    ready = runner.run_next()
    with pytest.raises(ValueError): runner.approve('local-fixture', 'wrong-fingerprint')
    launches = []
    monkeypatch.setattr(runner, '_start_lookahead', lambda: launches.append('one') or 1234)
    approved = runner.approve('local-fixture', ready['review_fingerprint'])
    assert approved['approved'] is True and launches == []
    assert runner.approve('local-fixture', ready['review_fingerprint'])['already_approved'] is True
    assert launches == []
    assert runner.queue.snapshot()['current'] is None


def test_automatic_child_reserves_one_configured_prepared_lookahead(pipeline_module, setup, monkeypatch):
    config, media, client, root = setup
    folder = root/'prepared/next-fixture'
    folder.mkdir(parents=True)
    (folder/'recipe.json').write_text('{}')
    (folder/'artwork.json').write_text(json.dumps({'slides': [{'path': '../../picture.png'}]*4}))
    config.update(preparation_dir=str(root/'prepared'), lookahead_slug='next-fixture')
    (root/'config.json').write_text(json.dumps(config))
    launches = []
    class Child:
        pid = 1234
    actual_popen = pipeline_module.subprocess.Popen
    def launch(command, **kwargs):
        if '--lookahead' not in command:
            return actual_popen(command, **kwargs)
        assert not runner.execution_lock.exists(), 'Child must be able to acquire runner lock immediately'
        launches.append(command)
        return Child()
    monkeypatch.setattr(pipeline_module.subprocess, 'Popen', launch)
    runner = pipeline_module.LessonPipeline(config, config_path=root/'config.json', media=media, client_factory=client)
    result = runner.run_next()
    assert result['status'] == 'ready'
    state = runner.queue.snapshot()
    assert state['current']['entry']['slug'] == 'local-fixture'
    assert state['lookahead']['entry']['slug'] == 'next-fixture'
    assert state['lookahead']['status'] == 'claimed'
    assert len(launches) == 1
    assert launches[0][-3:] == ['--next', '--lookahead', '--resume']
    assert runner.run_next()['preview_url'] == result['preview_url']
    assert len(launches) == 1 and len(runner.queue.snapshot()['jobs']) == 2


def test_execution_lock_blocks_another_process_without_an_extra_job(pipeline_module, setup):
    config, media, client, root = setup
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    runner.execution_lock.parent.mkdir(parents=True, exist_ok=True)
    runner.execution_lock.write_text('another-owner')
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next()
    assert client.posts == 0 and not (root/'state.json').exists()


def test_per_lesson_local_art_configuration_is_used_for_all_four_workers(pipeline_module, setup, monkeypatch):
    config, media, client, root = setup
    config['preparation_dir'] = str(root/'prepared')
    class UnconfiguredBase:
        def preflight(self): return {'ok': True}
        def make_image(self, *args): raise ValueError('Wrong unconfigured media instance')
    def factory(selected):
        return media if selected.get('recipe_file') else UnconfiguredBase()
    monkeypatch.setattr(pipeline_module, 'MediaService', factory)
    result = pipeline_module.LessonPipeline(config, client_factory=client).run_next()
    assert result['status'] == 'ready' and client.posts == 1


def test_service_inventory_never_claims_external_plan_services_are_free(pipeline_module):
    inventory = pipeline_module.service_inventory()
    assert inventory['paid_provider_apis_used'] is False
    assert inventory['local_tools_have_per_call_service_fee'] is False
    assert inventory['fully_standalone_image_generation'] is False
    assert {s['name'] for s in inventory['outside_services']} == {'AskLilOwl website / existing Render hosting', 'Existing ChatGPT built-in image tool'}
    assert all(s['confirmed_universally_free'] is False for s in inventory['outside_services'])


def test_post_success_followed_by_log_failure_stays_uncertain_without_repost(pipeline_module, setup, monkeypatch):
    config, media, client, _ = setup
    from lesson_platform.private_http import UploadUncertain
    original = pipeline_module._Timing.event
    def fail_after_post(self, stage, event, **fields):
        if stage == 'upload' and event == 'finish': raise OSError('Fixture log failure after POST')
        return original(self, stage, event, **fields)
    monkeypatch.setattr(pipeline_module._Timing, 'event', fail_after_post)
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    with pytest.raises(UploadUncertain): runner.run_next()
    assert runner.queue.snapshot()['current']['status'] == 'upload_uncertain'
    with pytest.raises(pipeline_module.PipelineBlocked): runner.run_next(resume=True)
    assert client.posts == 1


def test_ready_state_write_failure_reconciles_without_another_post(pipeline_module, setup, monkeypatch):
    config, media, client, _ = setup
    from lesson_platform.private_http import UploadUncertain
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    original = runner.queue.update
    failures = []
    def fail_ready(job_id, status, **fields):
        if status == 'ready' and not failures:
            failures.append('one')
            raise OSError('Fixture commit acknowledgement lost')
        return original(job_id, status, **fields)
    monkeypatch.setattr(runner.queue, 'update', fail_ready)
    with pytest.raises(UploadUncertain): runner.run_next()
    job = runner.queue.snapshot()['current']
    assert job['status'] == 'upload_uncertain'
    def stored_receipt(self, slug):
        return dict(slug=slug, fingerprint=job['review_fingerprint'], slides=4, audio_clips=4,
                    active=False, approved_for_publish=False, preview_url=f'/admin/published-lessons/{slug}/preview')
    monkeypatch.setattr(client, 'lookup', stored_receipt)
    assert runner.run_next(resume=True)['status'] == 'ready'
    assert client.posts == 1


def test_resume_returns_job_that_became_ready_before_execution_lock(pipeline_module, setup, monkeypatch):
    config, media, client, _ = setup
    runner = pipeline_module.LessonPipeline(config, media=media, client_factory=client)
    ready = runner.run_next()
    original = runner.queue.snapshot
    calls = []
    def stale_first():
        value = original()
        if not calls:
            calls.append('one')
            value['current']['status'] = 'preparing'
        return value
    monkeypatch.setattr(runner.queue, 'snapshot', stale_first)
    assert runner.run_next(resume=True)['preview_url'] == ready['preview_url']
    assert client.posts == 1
