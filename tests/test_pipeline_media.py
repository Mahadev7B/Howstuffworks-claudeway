"""Offline media tests use supplied recipes/artwork and saved audio fixtures."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import pytest
from PIL import Image, ImageDraw
from lesson_platform import pipeline_media as media


@pytest.fixture(name='media')
def media_fixture():
    from lesson_platform import pipeline_media
    return pipeline_media


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('Media must stay offline'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def picture(color='blue', size=(1024, 640), format='PNG'):
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 70, size[0]-80, size[1]-80), fill=color)
    draw.ellipse((130, 130, 300, 300), fill='yellow')
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def recipe():
    return dict(slides=[dict(title=f'Step {i}', fun_fact='A reviewed separate fact.',
                            image_prompt=f'A reviewed illustration of step {i}') for i in range(4)], source_issues=[])


@pytest.fixture
def local_art(tmp_path):
    source = tmp_path/'source.json'
    source.write_bytes(canonical([{'slug': 'fixture', 'slides': recipe()['slides']}]))
    slides = []
    for i, color in enumerate(['blue', 'red', 'green', 'orange']):
        path = tmp_path/f'art-{i}.png'
        path.write_bytes(picture(color))
        slides.append(dict(path=path.name, sha256=sha(path.read_bytes()),
                           prompt_sha256=sha(recipe()['slides'][i]['image_prompt'].encode())))
    manifest = tmp_path/'artwork.json'
    manifest.write_bytes(canonical(dict(source_manifest=source.name, source_manifest_sha256=sha(source.read_bytes()), slides=slides)))
    return manifest, source, slides


def test_preflight_reports_no_outside_services():
    report = media.MediaService({}).preflight()
    assert report['ok'] and report['estimated_cost_usd'] == 0 and report['outside_services'] == []


def test_missing_artwork_stops_without_outputs(tmp_path):
    with pytest.raises(ValueError, match='needs_builtin_artwork'):
        media.MediaService({}).make_image('An illustration', tmp_path/'output.png', 0)
    assert not list(tmp_path.iterdir())


def test_supplied_art_copies_exact_bytes_and_preserves_cache(tmp_path, local_art):
    manifest, _, slides = local_art
    service = media.MediaService({'artwork_manifest': str(manifest)})
    target = tmp_path/'output/slide.png'
    prompt = recipe()['slides'][0]['image_prompt']
    first = service.make_image(prompt, target, 0)
    assert first['cost_usd'] == 0 and target.read_bytes() == (tmp_path/slides[0]['path']).read_bytes()
    assert first['sha256'] == slides[0]['sha256']
    assert service.make_image(prompt, target, 0)['reused'] is True
    target.write_bytes(picture('red'))
    with pytest.raises(ValueError, match='changed'):
        service.make_image(prompt, target, 0)


@pytest.mark.parametrize('defect', ['source', 'art', 'prompt', 'missing'])
def test_art_requires_matching_prompt_source_and_bytes(tmp_path, local_art, defect):
    manifest, source, slides = local_art
    prompt = recipe()['slides'][0]['image_prompt']
    if defect == 'source': source.write_bytes(b'changed source')
    elif defect == 'art': (tmp_path/slides[0]['path']).write_bytes(picture('red'))
    elif defect == 'prompt': prompt = 'Different prompt'
    else: (tmp_path/slides[0]['path']).unlink()
    target = tmp_path/'output.png'
    with pytest.raises(ValueError):
        media.MediaService({'artwork_manifest': str(manifest)}).make_image(prompt, target, 0)
    assert not target.exists()


def test_decoded_checks_detect_structural_defects(tmp_path):
    files = []
    for name, data in [('a.png', picture()), ('copy.png', picture()),
                       ('small.png', picture(size=(320, 320))), ('invalid.png', b'not-image')]:
        path = tmp_path/name
        path.write_bytes(data)
        files.append(path)
    Image.new('RGB', (1024, 640), 'white').save(tmp_path/'uniform.png')
    files.append(tmp_path/'uniform.png')
    report = media.MediaService({}).check_images(files)
    flags = {flag for result in report['images'] for flag in result['flags']}
    assert not report['ok']
    assert {'duplicate', 'wrong_aspect', 'wrong_dimensions', 'decode_failed', 'blank_or_near_uniform'} <= flags
    assert any('anatomy' in line for line in report['limitations'])


def test_existing_landscape_sizes_and_wrong_formats(tmp_path):
    a, b, c = tmp_path/'a.png', tmp_path/'b.png', tmp_path/'large.png'
    a.write_bytes(picture())
    b.write_bytes(picture(format='JPEG'))
    c.write_bytes(picture('red', size=(1536, 1024)))
    report = media.MediaService({}).check_images([a, b, c])
    assert {'wrong_format', 'near_duplicate'} <= set(report['images'][1]['flags'])
    assert not report['images'][2]['flags']


def test_reviewed_inline_recipe_preserves_originals_and_caches(tmp_path):
    lines = ['A. ', 'B.', 'C.', 'D.']
    entry = dict(slug='fixture', recipe_reviewed=True, recipe=recipe())
    original = canonical(entry)
    service = media.MediaService({})
    result = service.prepare_recipe(entry, lines, tmp_path)
    assert result == recipe() and canonical(entry) == original and lines[0] == 'A. '
    assert service.prepare_recipe(entry, lines, tmp_path) == result
    with pytest.raises(ValueError, match='changed'):
        service.prepare_recipe(entry, ['Changed', 'B.', 'C.', 'D.'], tmp_path)


@pytest.mark.parametrize('entry', [{'slug': 'x'}, {'recipe': recipe()},
                                  {'recipe_reviewed': True, 'recipe': {'slides': [], 'source_issues': []}}])
def test_missing_reviewed_recipes_never_create_filler(tmp_path, entry):
    with pytest.raises(ValueError):
        media.MediaService({}).prepare_recipe(entry, ['A', 'B', 'C', 'D'], tmp_path)
    assert not list(tmp_path.iterdir())


def test_recipe_cannot_edit_original_narration(tmp_path):
    supplied = recipe()
    supplied['slides'][0]['narration'] = 'Rewritten'
    with pytest.raises(ValueError):
        media.MediaService({}).prepare_recipe({'recipe_reviewed': True, 'recipe': supplied}, ['A', 'B', 'C', 'D'], tmp_path)


def test_local_recipe_file_requires_review_source_binding_and_immutable_cache(tmp_path):
    lines = ['A', 'B', 'C', 'D']
    path = tmp_path/'supplied.json'
    envelope = dict(reviewed=True, original_lines_sha256=sha(canonical(lines)), recipe=recipe())
    path.write_bytes(canonical(envelope))
    service = media.MediaService({'recipe_file': str(path)})
    assert service.prepare_recipe({'slug': 'fixture'}, lines, tmp_path/'job') == recipe()
    envelope['recipe']['slides'][0]['fun_fact'] = 'Changed separate fact.'
    path.write_bytes(canonical(envelope))
    with pytest.raises(ValueError, match='changed'):
        service.prepare_recipe({'slug': 'fixture'}, lines, tmp_path/'job')
    envelope['reviewed'] = False
    path.write_bytes(canonical(envelope))
    with pytest.raises(ValueError, match='reviewed'):
        service.prepare_recipe({'slug': 'fixture'}, lines, tmp_path/'other')


@pytest.mark.parametrize('defect', ['missing', 'held', 'lines', 'source_hash', 'recipe_bytes', 'issue', 'correction', 'schema'])
def test_required_factual_cache_blocks_missing_or_changed_review_before_any_output(tmp_path, defect):
    lines = ['A', 'B', 'C', 'D']
    path = tmp_path/'recipe.json'
    path.write_bytes(canonical(dict(reviewed=True, original_lines_sha256=sha(canonical(lines)), recipe=recipe())))
    review = dict(schema_version=1, status='passed', original_lines=lines,
                  original_lines_sha256=sha(canonical(lines)), recipe_sha256=sha(path.read_bytes()),
                  source_issues=[], required_corrections=[])
    if defect == 'held': review['status'] = 'held'
    elif defect == 'lines': review['original_lines'] = ['Changed', 'B', 'C', 'D']
    elif defect == 'source_hash': review['original_lines_sha256'] = '0'*64
    elif defect == 'recipe_bytes': path.write_bytes(path.read_bytes()+b'\n')
    elif defect == 'issue': review['source_issues'] = ['Unresolved source claim']
    elif defect == 'correction': review['required_corrections'] = ['Needs permission']
    elif defect == 'schema': review['schema_version'] = True
    if defect != 'missing': (tmp_path/'review.json').write_bytes(canonical(review))
    with pytest.raises(ValueError, match='review_cache'):
        media.MediaService({'recipe_file': str(path), 'require_review_cache': True}).prepare_recipe(
            {'slug': 'fixture'}, lines, tmp_path/'job')
    assert not (tmp_path/'job').exists()


def test_required_review_cache_is_revalidated_even_when_recipe_is_already_cached(tmp_path):
    lines = ['A', 'B', 'C', 'D']
    path = tmp_path/'recipe.json'
    path.write_bytes(canonical(dict(reviewed=True, original_lines_sha256=sha(canonical(lines)), recipe=recipe())))
    review = dict(schema_version=1, status='passed', original_lines=lines,
                  original_lines_sha256=sha(canonical(lines)), recipe_sha256=sha(path.read_bytes()),
                  source_issues=[], required_corrections=[])
    review_path = tmp_path/'review.json'
    review_path.write_bytes(canonical(review))
    service = media.MediaService({'recipe_file': str(path), 'require_review_cache': True})
    assert service.prepare_recipe({'slug': 'fixture'}, lines, tmp_path/'job') == recipe()
    review['status'] = 'held'
    review_path.write_bytes(canonical(review))
    with pytest.raises(ValueError, match='review_cache'):
        service.prepare_recipe({'slug': 'fixture'}, lines, tmp_path/'job')



def test_audio_rejects_manifest_that_would_strip_originals_before_running(media, tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'slug': 'fixture', 'slides': [dict(narration=' A ', explanation=' A ', fun_fact='B')]*4}]))
    monkeypatch.setattr(media.subprocess, 'run', lambda *a, **k: pytest.fail('Invalid source reached generation'))
    with pytest.raises(ValueError, match='whitespace|verbatim'):
        media.MediaService({}).make_audio(manifest, tmp_path/'audio')


def test_audio_cache_rejects_stale_receipt_without_regeneration(media, tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'slug': 'fixture', 'slides': [dict(narration='A', explanation='A', fun_fact='B')]*4}]))
    folder = tmp_path/'audio'; folder.mkdir()
    (folder/'receipt.json').write_text(json.dumps({'slug': 'fixture', 'source_manifest_sha256': 'bad', 'clips': []}))
    monkeypatch.setattr(media.subprocess, 'run', lambda *a, **k: pytest.fail('Stale cache regenerated'))
    with pytest.raises(ValueError, match='cache|changed|stale'):
        media.MediaService({}).make_audio(manifest, folder)


def test_audio_decodes_existing_four_mp3s_and_reuses_without_tts(media, tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[2]
    fixture = workspace/'Howstuffworks-claudeway-library/tmp/review-packets/digestion14-kokoro-v1'
    manifest = workspace/'Howstuffworks-claudeway-library/assets/editorial/lesson-14-digestion.json'
    folder = tmp_path/'audio'
    shutil.copytree(fixture/'raw', folder/'raw')
    shutil.copytree(fixture/'normalized', folder/'normalized')
    for clip in json.loads((fixture/'receipt.json').read_text())['clips']:
        target = folder/'normalized'/f"slide-{clip['slide_index']+1:02d}-kokoro-af-heart-v1.mp3"
        target.with_suffix('.normalization-receipt.json').write_text(json.dumps({
            'raw_sha256': clip['output_sha256'], 'normalized_sha256': clip['normalized_sha256'],
            'filter': 'loudnorm=I=-16:TP=-1.5:LRA=11'}))
    service = media.MediaService({})
    original_run = service._run
    def only_decode(args, temp):
        assert service.paths['kokoro_generator'] not in args, 'Existing fixtures must not regenerate TTS'
        assert '-codec:a' not in args, 'Existing fixtures must only decode/measure'
        return original_run(args, temp)
    monkeypatch.setattr(service, '_run', only_decode)
    result = service.make_audio(manifest, folder)
    assert len(result['clips']) == 4 and result['human_listening'] == 'pending'
    assert 45 < result['total_seconds'] < 65
    assert all(c['sample_rate'] == 24000 and c['channels'] == 1 and c['finite'] for c in result['clips'])
    assert all(c['speed'] == 1 and -17.2 <= c['integrated_lufs'] <= -15.5 and -10 <= c['true_peak_dbtp'] <= -1 for c in result['clips'])
    monkeypatch.setattr(service, '_run', lambda *args: pytest.fail('Valid receipt must not repeat any subprocess'))
    assert service.make_audio(manifest, folder) == result
    raw = folder/'normalized/slide-01-kokoro-af-heart-v1.mp3'
    raw.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        service.make_audio(manifest, folder)


def test_local_commands_get_temp_and_utf8_from_process_start(media, tmp_path, monkeypatch):
    def run(args, **kwargs):
        assert kwargs['env']['TEMP'] == str((tmp_path/'temp').resolve())
        assert kwargs['env']['TMP'] == str((tmp_path/'temp').resolve())
        assert kwargs['env']['PYTHONUTF8'] == '1'
        assert kwargs['env']['PYTHONIOENCODING'] == 'utf-8'
        return 'fixture result'
    monkeypatch.setattr(media.subprocess, 'run', run)
    assert media.MediaService({})._run(['fixture-command'], tmp_path/'temp') == 'fixture result'


def test_unbound_normalized_audio_is_not_silently_accepted(media, tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[2]
    fixture = workspace/'Howstuffworks-claudeway-library/tmp/review-packets/digestion14-kokoro-v1'
    manifest = workspace/'Howstuffworks-claudeway-library/assets/editorial/lesson-14-digestion.json'
    folder = tmp_path/'audio'
    shutil.copytree(fixture/'raw', folder/'raw')
    shutil.copytree(fixture/'normalized', folder/'normalized')
    service = media.MediaService({})
    with pytest.raises(ValueError, match='normaliz|binding'):
        service.make_audio(manifest, folder)
    assert not (folder/'receipt.json').exists()


@pytest.mark.parametrize('change', ['voice', 'engine', 'normalization', 'player_rate'])
def test_audio_receipt_rejects_unsupported_voice_engine_or_double_slowing(media, tmp_path, change):
    # A real recorded receipt with copied artifacts avoids any synthesis.
    workspace = Path(__file__).resolve().parents[2]
    fixture = workspace/'Howstuffworks-claudeway-library/tmp/review-packets/digestion14-kokoro-v1'
    manifest = workspace/'Howstuffworks-claudeway-library/assets/editorial/lesson-14-digestion.json'
    folder = tmp_path/'audio'
    shutil.copytree(fixture/'raw', folder/'raw')
    shutil.copytree(fixture/'normalized', folder/'normalized')
    receipt = json.loads((fixture/'receipt.json').read_text())
    service = media.MediaService({})
    receipt['source_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt['runtime_hashes'] = {key+'_sha256': service._file_hash(path) for key, path in service.paths.items()}
    receipt['normalization'] = 'loudnorm=I=-16:TP=-1.5:LRA=11; no tempo filter'
    if change in ('voice', 'engine'):
        receipt['clips'][0][change] = 'wrong'
    elif change == 'normalization':
        receipt['normalization'] = 'atempo=0.9,loudnorm=I=-16:TP=-1.5:LRA=11'
    else:
        receipt['player_rate'] = 0.81
    (folder/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        service.make_audio(manifest, folder)


def test_audio_generation_commands_bind_exact_source_and_apply_no_tempo_filter(media, tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[2]
    fixture = workspace/'Howstuffworks-claudeway-library/tmp/review-packets/digestion14-kokoro-v1'
    manifest = workspace/'Howstuffworks-claudeway-library/assets/editorial/lesson-14-digestion.json'
    folder = tmp_path/'audio'
    service = media.MediaService({})
    real_run = media.subprocess.run
    generated, normalized = [], []
    def fake_generation(args, **kwargs):
        assert kwargs['env']['TEMP'] == str(folder/'temp')
        assert kwargs['env']['TMP'] == str(folder/'temp')
        assert kwargs['env']['PYTHONUTF8'] == '1'
        if len(args) > 1 and args[1] == str(service.paths['kokoro_generator']):
            target = Path(args[5]); target.parent.mkdir(parents=True, exist_ok=True)
            assert args[2] == str(manifest)
            assert args[3] == 'how-your-digestive-system-works'
            assert args[6:] == ['--model', str(service.paths['kokoro_model']), '--voices', str(service.paths['kokoro_voices'])]
            shutil.copyfile(fixture/'raw'/target.name, target)
            shutil.copyfile((fixture/'raw'/target.name).with_suffix('.json'), target.with_suffix('.json'))
            generated.append(int(args[4]))
            return media.subprocess.CompletedProcess(args, 0, '', '')
        if '-codec:a' in args:
            assert args[args.index('-af')+1] == 'loudnorm=I=-16:TP=-1.5:LRA=11'
            assert args[args.index('-ar')+1] == '24000' and args[args.index('-ac')+1] == '1'
            assert '-n' in args and '-y' not in args
            target = Path(args[-1])
            shutil.copyfile(fixture/'normalized'/target.name, target)
            normalized.append(target.name)
            return media.subprocess.CompletedProcess(args, 0, '', '')
        return real_run(args, **kwargs)
    monkeypatch.setattr(media.subprocess, 'run', fake_generation)
    result = service.make_audio(manifest, folder)
    assert generated == [1, 2, 3, 4] and len(normalized) == 4
    assert len(result['clips']) == 4
    assert len(list((folder/'normalized').glob('*.normalization-receipt.json'))) == 4


@pytest.fixture
def loudness_fixture(tmp_path, monkeypatch):
    """Keep real MP3 decoding/filtering; inject one measured borderline result."""
    workspace = Path(__file__).resolve().parents[2]
    fixture = workspace/'Howstuffworks-claudeway-library/tmp/review-packets/digestion14-kokoro-v1'
    manifest = workspace/'Howstuffworks-claudeway-library/assets/editorial/lesson-14-digestion.json'
    folder = tmp_path/'audio'
    shutil.copytree(fixture/'raw', folder/'raw')
    shutil.copytree(fixture/'normalized', folder/'normalized')
    for clip in json.loads((fixture/'receipt.json').read_text())['clips']:
        target = folder/'normalized'/f"slide-{clip['slide_index']+1:02d}-kokoro-af-heart-v1.mp3"
        target.with_suffix('.normalization-receipt.json').write_bytes(canonical(dict(
            raw_sha256=clip['output_sha256'], normalized_sha256=clip['normalized_sha256'], filter=media.NORMALIZATION)))
    return media.MediaService({}), manifest, folder


@pytest.mark.parametrize('failed_lufs', [-17.21, -17.26])
def test_lufs_only_failure_gets_one_measured_retry_and_preserves_failed_bytes(loudness_fixture, monkeypatch, failed_lufs):
    service, manifest, folder = loudness_fixture
    target = folder/'normalized/slide-01-kokoro-af-heart-v1.mp3'
    binding = target.with_suffix('.normalization-receipt.json')
    original, original_binding = target.read_bytes(), binding.read_bytes()
    real_run = service._run
    filtered, injected = [], []
    def run(args, temp):
        assert service.paths['kokoro_generator'] not in args, 'Fallback must not regenerate TTS'
        result = real_run(args, temp)
        if '-codec:a' in args:
            filtered.append([str(arg) for arg in args])
        if '-f' in args and target in args and not injected:
            start, end = result.stderr.rfind('{'), result.stderr.rfind('}')+1
            values = json.loads(result.stderr[start:end])
            values['input_i'] = str(failed_lufs)
            result.stderr = result.stderr[:start]+json.dumps(values)+result.stderr[end:]
            injected.append(True)
        return result
    monkeypatch.setattr(service, '_run', run)
    result = service.make_audio(manifest, folder)
    fallback = result['clips'][0]['normalization_fallback']
    assert fallback['failed_qa']['integrated_lufs'] == failed_lufs
    assert Path(fallback['failed_output']).read_bytes() == original
    assert Path(fallback['failed_binding']).read_bytes() == original_binding
    assert fallback['failed_output_sha256'] == sha(original)
    assert fallback['failed_binding_sha256'] == sha(original_binding)
    assert fallback['raw_sha256'] == sha((folder/'raw'/target.name).read_bytes())
    assert len(filtered) == 1
    applied = filtered[0][filtered[0].index('-af')+1]
    assert applied == fallback['filter'] and applied.startswith('loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=')
    assert ':linear=true' in applied and 'atempo' not in applied
    assert '-n' in filtered[0] and '-y' not in filtered[0]
    assert result['clips'][0]['normalized_sha256'] == sha(target.read_bytes())
    assert -17.2 <= result['clips'][0]['integrated_lufs'] <= -15.5
    assert -10 <= result['clips'][0]['true_peak_dbtp'] <= -1
    # A later slide can fail before the lesson receipt is saved. Resuming must
    # reuse the repaired clip and its provenance, with no second encode.
    (folder/'receipt.json').unlink()
    assert service.make_audio(manifest, folder) == result
    assert len(filtered) == 1
    monkeypatch.setattr(service, '_run', lambda *args: pytest.fail('Matching receipt must not repeat normalization'))
    assert service.make_audio(manifest, folder) == result


@pytest.mark.parametrize('change', [dict(finite=False), dict(sample_rate=22050), dict(channels=2),
                                  dict(duration_seconds=181), dict(peak=1.0), dict(peak=0),
                                  dict(input_tp='-0.5'), dict(input_i='-inf')])
def test_lufs_retry_does_not_mask_another_audio_failure(loudness_fixture, monkeypatch, change):
    service, manifest, folder = loudness_fixture
    target = folder/'normalized/slide-01-kokoro-af-heart-v1.mp3'
    before = target.read_bytes()
    real_run = service._run
    def run(args, temp):
        assert '-codec:a' not in args, 'Other audio failures must not trigger a retry'
        result = real_run(args, temp)
        if target in args and '-c' in args:
            values = json.loads(result.stdout)
            values.update({k: v for k, v in change.items() if not k.startswith('input_')})
            result.stdout = json.dumps(values)
        if target in args and '-f' in args:
            start, end = result.stderr.rfind('{'), result.stderr.rfind('}')+1
            values = json.loads(result.stderr[start:end])
            values.update(input_i='-17.26')
            values.update({k: v for k, v in change.items() if k.startswith('input_')})
            result.stderr = result.stderr[:start]+json.dumps(values)+result.stderr[end:]
        return result
    monkeypatch.setattr(service, '_run', run)
    with pytest.raises(ValueError): service.make_audio(manifest, folder)
    assert target.read_bytes() == before
    assert not list((folder/'normalized').glob('*two-pass*'))
    assert not (folder/'receipt.json').exists()


def test_failed_two_pass_stops_and_restart_does_not_retry_again(loudness_fixture, monkeypatch):
    service, manifest, folder = loudness_fixture
    target = folder/'normalized/slide-01-kokoro-af-heart-v1.mp3'
    before = target.read_bytes()
    real_run = service._run
    attempts = []
    def run(args, temp):
        result = real_run(args, temp)
        if '-codec:a' in args: attempts.append(True)
        if '-f' in args and Path(args[args.index('-i')+1]).parent == folder/'normalized':
            start, end = result.stderr.rfind('{'), result.stderr.rfind('}')+1
            values = json.loads(result.stderr[start:end]); values['input_i'] = '-17.26'
            result.stderr = result.stderr[:start]+json.dumps(values)+result.stderr[end:]
        return result
    monkeypatch.setattr(service, '_run', run)
    with pytest.raises(ValueError): service.make_audio(manifest, folder)
    assert len(attempts) == 1
    assert target.read_bytes() == before
    assert not (folder/'receipt.json').exists()
    with pytest.raises((ValueError, FileExistsError)): service.make_audio(manifest, folder)
    assert len(attempts) == 1


def test_cube_audio_prepares_sixteen_saved_clips_without_regeneration(tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[2]
    fixture = workspace/'Howstuffworks-claudeway-library/tmp/review-packets/digestion14-kokoro-v1'
    original_manifest = workspace/'Howstuffworks-claudeway-library/assets/editorial/lesson-14-digestion.json'
    record = json.loads(original_manifest.read_text())[0]
    record.update(slug='solve-a-2x2-rubiks-cube', sort_order=28)
    record['slides'] = [dict(record['slides'][0]) for _ in range(16)]
    manifest = tmp_path/'cube.json'
    manifest.write_text(json.dumps([record]))
    folder = tmp_path/'audio'
    (folder/'raw').mkdir(parents=True)
    (folder/'normalized').mkdir()
    raw = fixture/'raw/slide-01-kokoro-af-heart-v1.mp3'
    normalized = fixture/'normalized'/raw.name
    meta = json.loads(raw.with_suffix('.json').read_text())
    for i in range(16):
        target = folder/'raw'/f'slide-{i+1:02d}-kokoro-af-heart-v1.mp3'
        target.write_bytes(raw.read_bytes())
        target.with_suffix('.json').write_text(json.dumps(dict(meta, lesson=record['slug'], slide_index=i)))
        output = folder/'normalized'/target.name
        output.write_bytes(normalized.read_bytes())
        output.with_suffix('.normalization-receipt.json').write_text(json.dumps(dict(
            raw_sha256=sha(raw.read_bytes()), normalized_sha256=sha(normalized.read_bytes()), filter=media.NORMALIZATION)))
    service = media.MediaService({})
    real_run = service._run
    def decode_only(args, temp):
        assert service.paths['kokoro_generator'] not in args
        assert '-codec:a' not in args
        return real_run(args, temp)
    monkeypatch.setattr(service, '_run', decode_only)
    result = service.make_audio(manifest, folder)
    assert len(result['clips']) == 16
    assert [clip['slide_index'] for clip in result['clips']] == list(range(16))
    assert all(clip['lesson'] == record['slug'] and clip['voice'] == 'af_heart' and clip['speed'] == 1 for clip in result['clips'])
    monkeypatch.setattr(service, '_run', lambda *args: pytest.fail('Matching cube cache repeated subprocesses'))
    assert service.make_audio(manifest, folder) == result


@pytest.mark.parametrize('slug,sort_order,count', [('another-topic', 28, 16),
                                                 ('solve-a-2x2-rubiks-cube', 29, 16),
                                                 ('solve-a-2x2-rubiks-cube', 28, 4)])
def test_audio_sixteen_slide_exception_cannot_change_other_lesson_shapes(tmp_path, monkeypatch, slug, sort_order, count):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([dict(slug=slug, sort_order=sort_order,
        slides=[dict(narration='A', explanation='A', fun_fact='B')]*count)]))
    service = media.MediaService({})
    monkeypatch.setattr(service, '_run', lambda *args: pytest.fail('Invalid shape reached audio commands'))
    with pytest.raises(ValueError): service.make_audio(manifest, tmp_path/'audio')
