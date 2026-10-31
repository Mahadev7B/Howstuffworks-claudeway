"""One local command replaces the copied build/check scripts; no production writes."""
import hashlib
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image


@pytest.fixture
def builder():
    assert importlib.util.find_spec('lesson_platform.private_package'), 'reusable private package builder missing'
    return importlib.import_module('lesson_platform.private_package')


@pytest.fixture
def inputs(tmp_path):
    bundled = Path(__file__).parents[2] / 'japan-short/vendor/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    decoder = shutil.which('ffmpeg') or str(bundled)
    assert Path(decoder).is_file(), 'Tests require the existing FFmpeg, never install it'
    audio = subprocess.run([decoder, '-nostdin', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'anullsrc=r=24000:cl=mono', '-t', '0.4', '-c:a', 'libmp3lame',
                            '-f', 'mp3', 'pipe:1'], capture_output=True, check=True).stdout
    Image.new('RGB', (640, 400), '#b9d9cb').save(tmp_path / 'picture.png')
    originals = ['First original line.', 'Second original line.', 'Third original line.', 'Fourth original line.']
    draft = tmp_path / 'draft.md'
    draft.write_text('# Draft\n\n## Fixture topic\n\n' + '\n'.join(f'{i+1}. {s}' for i, s in enumerate(originals)), encoding='utf-8')
    slides = [dict(title=f'Step {i+1}', explanation=s, narration=s, fun_fact='Separate fact.', image='picture.png') for i, s in enumerate(originals)]
    entry = dict(slug='local-fixture', question='A local question?', title='Fixture topic', summary='Not for upload.',
                 category='Science', sort_order=99, featured_rank=None, cover='picture.png', slides=slides)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([entry]), encoding='utf-8')
    audio_dir = tmp_path / 'audio'
    (audio_dir / 'normalized').mkdir(parents=True)
    clips = []
    for i, s in enumerate(originals):
        script = s + ' Fun fact: Separate fact.'
        (audio_dir / 'normalized' / f'slide-{i+1:02d}-kokoro-af-heart-v1.mp3').write_bytes(audio)
        clips.append(dict(slide_index=i, lesson='local-fixture', script=script,
                          source_text_hash=hashlib.sha256(script.encode()).hexdigest(), normalized_sha256=hashlib.sha256(audio).hexdigest(),
                          engine='Kokoro-82M ONNX v1.0', voice='af_heart', speed=1.0, finite=True,
                          sample_rate=24000, channels=1, integrated_lufs=-16.5, true_peak_dbtp=-1.3))
    (audio_dir / 'receipt.json').write_text(json.dumps(dict(slug='local-fixture', clips=clips)), encoding='utf-8')
    return dict(manifest=manifest, draft=draft, topic='Fixture topic', audio_dir=audio_dir,
                output_dir=tmp_path / 'checked', ffmpeg=Path(decoder)), entry


def test_builds_exact_upload_contract_without_storage(builder, inputs, monkeypatch):
    args, _ = inputs
    from lesson_platform import library
    monkeypatch.setattr(library, '_connect', lambda: pytest.fail('Local packaging must not connect to DB'))
    report = builder.build_package(**args)
    packet = json.loads((args['output_dir'] / 'stage-packet.json').read_bytes())
    assert report['ok'] is True and report['reused'] is False
    assert packet['entry']['slug'] == 'local-fixture' and packet['player_rate'] == 0.9
    assert len(packet['clips']) == 4
    assert packet['clips'][2]['script'] == 'Third original line. Fun fact: Separate fact.'
    assert packet['clips'][2]['voice'] == 'af_heart'
    assert 'active' not in packet['entry'] and 'approved_for_publish' not in packet['entry']
    assert hashlib.sha256((args['output_dir'] / 'stage-packet.json').read_bytes()).hexdigest() == report['packet_sha256']


@pytest.mark.parametrize('defect', ['draft', 'script', 'hash', 'voice', 'speed', 'missing', 'clip_count', 'loudness', 'invalid_mp3',
                                  'manifest_shape', 'receipt_shape', 'metadata_shape', 'whitespace'])
def test_bad_inputs_fail_before_creating_output(builder, inputs, defect):
    args, _ = inputs
    receipt_path = args['audio_dir'] / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if defect == 'draft': args['draft'].write_text('## Fixture topic\n1. Different line.\n2. Two.\n3. Three.\n4. Four.\n')
    elif defect == 'script': receipt['clips'][0]['script'] = 'Wrong words.'
    elif defect == 'hash': receipt['clips'][0]['normalized_sha256'] = '0'*64
    elif defect == 'voice': receipt['clips'][0]['voice'] = 'marin'
    elif defect == 'speed': receipt['clips'][0]['speed'] = 0.9
    elif defect == 'missing': (args['audio_dir'] / 'normalized/slide-01-kokoro-af-heart-v1.mp3').rename(args['audio_dir'] / 'original.mp3')
    elif defect == 'clip_count': receipt['clips'].pop()
    elif defect == 'loudness': receipt['clips'][0]['integrated_lufs'] = -35.0
    elif defect == 'invalid_mp3':
        bad = b'not an mp3'
        (args['audio_dir'] / 'normalized/slide-01-kokoro-af-heart-v1.mp3').write_bytes(bad)
        receipt['clips'][0]['normalized_sha256'] = hashlib.sha256(bad).hexdigest()
    elif defect == 'manifest_shape': args['manifest'].write_text('[null]')
    elif defect == 'receipt_shape': receipt = []
    elif defect == 'metadata_shape': receipt['clips'][0] = None
    elif defect == 'whitespace':
        args['draft'].write_text(args['draft'].read_text().replace('1. First original line.', '1. First original line.  '))
        entries = json.loads(args['manifest'].read_text())
        entries[0]['slides'][0]['narration'] += '  '
        entries[0]['slides'][0]['explanation'] += '  '
        args['manifest'].write_text(json.dumps(entries))
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, OSError)):
        builder.build_package(**args)
    assert not args['output_dir'].exists()


def test_identical_rerun_reuses_and_different_output_is_preserved(builder, inputs):
    args, _ = inputs
    builder.build_package(**args)
    packet_path = args['output_dir'] / 'stage-packet.json'
    original = packet_path.read_bytes()
    before = packet_path.stat().st_mtime_ns
    assert builder.build_package(**args)['reused'] is True
    assert packet_path.stat().st_mtime_ns == before
    packet_path.write_bytes(b'preserve this other package')
    with pytest.raises(ValueError): builder.build_package(**args)
    assert packet_path.read_bytes() == b'preserve this other package'


def test_missing_decoder_fails_closed(builder, inputs):
    args, _ = inputs
    args['ffmpeg'] = args['output_dir'].parent / 'absent-ffmpeg.exe'
    with pytest.raises((ValueError, OSError, RuntimeError)): builder.build_package(**args)
    assert not args['output_dir'].exists()


def test_output_link_is_rejected_before_resolving_or_publishing(builder, inputs, monkeypatch):
    args, _ = inputs
    target = args['output_dir']
    # Only the filesystem's link-query boundary is replaced (Windows cannot
    # create symlinks without privileges on every test host).
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda p: p == target or original(p))
    with pytest.raises(ValueError): builder.build_package(**args)
    assert not target.exists()


def test_incomplete_existing_receipt_is_preserved_but_not_accepted(builder, inputs):
    args, _ = inputs
    report = builder.build_package(**args)
    receipt_path = args['output_dir'] / 'packet-receipt.json'
    incomplete = json.dumps({'packet_sha256': report['packet_sha256']})
    receipt_path.write_text(incomplete)
    with pytest.raises(ValueError): builder.build_package(**args)
    assert receipt_path.read_text() == incomplete


@pytest.fixture
def cube_inputs(inputs):
    args, entry = inputs
    slug = 'solve-a-2x2-rubiks-cube'
    lines = [f'Original cube step {i+1}.' for i in range(16)]
    entry.update(slug=slug, sort_order=28)
    entry['slides'] = [dict(entry['slides'][0], title=f'Step {i+1}', narration=line, explanation=line)
                       for i, line in enumerate(lines)]
    args['manifest'].write_text(json.dumps([entry]))
    args['draft'].write_text('## Fixture topic\n\n'+'\n'.join(f'{i+1}. {line}' for i, line in enumerate(lines)))
    receipt_path = args['audio_dir']/'receipt.json'
    original = json.loads(receipt_path.read_text())['clips'][0]
    audio = (args['audio_dir']/'normalized/slide-01-kokoro-af-heart-v1.mp3').read_bytes()
    clips = []
    for i, line in enumerate(lines):
        script = line+' Fun fact: Separate fact.'
        clips.append(dict(deepcopy(original), slide_index=i, lesson=slug, script=script,
                          source_text_hash=hashlib.sha256(script.encode()).hexdigest()))
        (args['audio_dir']/'normalized'/f'slide-{i+1:02d}-kokoro-af-heart-v1.mp3').write_bytes(audio)
    receipt_path.write_text(json.dumps(dict(slug=slug, clips=clips)))
    return args


def test_cube_package_preserves_all_sixteen_originals_and_clips(builder, cube_inputs):
    report = builder.build_package(**cube_inputs)
    packet = json.loads((cube_inputs['output_dir']/'stage-packet.json').read_bytes())
    assert report['slides'] == report['audio_clips'] == 16
    assert len(packet['entry']['lesson']['slides']) == len(packet['clips']) == 16
    assert packet['clips'][15]['script'] == 'Original cube step 16. Fun fact: Separate fact.'
    assert builder.build_package(**cube_inputs)['reused'] is True


@pytest.mark.parametrize('defect', ['other_slug', 'wrong_sort', 'missing_clip', 'last_source', 'last_audio'])
def test_cube_package_requires_exact_policy_source_and_complete_audio(builder, cube_inputs, defect):
    args = cube_inputs
    records = json.loads(args['manifest'].read_text())
    receipt_path = args['audio_dir']/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if defect == 'other_slug': records[0]['slug'] = 'different-sixteen-step-topic'
    elif defect == 'wrong_sort': records[0]['sort_order'] = 29
    elif defect == 'missing_clip': receipt['clips'].pop()
    elif defect == 'last_source': records[0]['slides'][15]['narration'] = 'Rewritten final line.'
    else: (args['audio_dir']/'normalized/slide-16-kokoro-af-heart-v1.mp3').unlink()
    args['manifest'].write_text(json.dumps(records))
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, OSError)): builder.build_package(**args)
    assert not args['output_dir'].exists()


def test_cli_from_another_directory_uses_own_app_root_and_reports_failure(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/build_private_lesson_package.py'
    assert script.is_file(), 'reusable CLI missing'
    result = subprocess.run([__import__('sys').executable, str(script), '--manifest', str(tmp_path/'missing.json'),
                             '--draft', str(tmp_path/'draft.md'), '--topic', 'Fixture', '--audio-dir', str(tmp_path),
                             '--output-dir', str(tmp_path/'checked'), '--ffmpeg', str(tmp_path/'missing.exe')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(result.stdout)['ok'] is False
    assert not (tmp_path/'checked').exists()
