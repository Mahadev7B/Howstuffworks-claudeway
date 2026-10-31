import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def module():
    assert importlib.util.find_spec('lesson_platform.cube_lesson') is not None, 'private cube animation binding missing'
    from lesson_platform import cube_lesson
    return cube_lesson

def lesson():
    return {'title':'Cube', 'slides':[dict(title=f'Step {i+1}',narration='Original body.',
        explanation='Original body.',fun_fact='Separate fact.') for i in range(16)]}

@pytest.fixture(autouse=True)
def authored_fixture(monkeypatch):
    # Keep the actual authored states/validator; bind generic non-private prose
    # so these tests run in any clone, without a sibling editorial workspace.
    m=module();path=ROOT/'lesson_platform/cube_animation_v2.json'
    data=json.loads(path.read_bytes())
    data['script_hashes']=['1779ffc68b1b3d752e98f7ee13af401590337c4dfc7b2ffc043b2a9a2a185471']*16
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:json.dumps(data).encode() if p==path else original(p))
    m._authored_data.cache_clear()
    yield
    m._authored_data.cache_clear()

def test_only_complete_matching_clean_cube_gets_all_sixteen_animations():
    data = module().private_cube_animation('solve-a-2x2-rubiks-cube',lesson())
    assert len(data['scenes']) == 16 and len(data['script_hashes']) == 16
    assert data['scenes'][7]['moves'] == ["R'", "D'", 'R', 'D']

def test_wrong_lesson_incomplete_or_changed_words_never_get_cube_animations():
    m=module()
    assert m.private_cube_animation('other-lesson',lesson()) is None
    for defect in ('missing','words','asterisk','fact'):
        value=lesson()
        if defect=='missing': value['slides'].pop()
        elif defect=='fact': value['slides'][0]['fun_fact']='Different fact.'
        else: value['slides'][0]['narration'] += '*' if defect=='asterisk' else ' changed'
        assert m.private_cube_animation('solve-a-2x2-rubiks-cube',value) is None

def test_private_binding_is_independent_and_does_not_mutate_saved_lesson():
    value=lesson();before=copy.deepcopy(value);m=module()
    first=m.private_cube_animation('solve-a-2x2-rubiks-cube',value)
    first['scenes'][0]['initial'].clear()
    assert len(m.private_cube_animation('solve-a-2x2-rubiks-cube',value)['scenes'][0]['initial']) == 24
    assert value==before

def _render(data):
    from flask import Flask, render_template
    from lesson_platform.library import with_publication_scripts
    app=Flask(__name__,template_folder=str(ROOT/'templates'))
    app.add_url_rule('/','home',lambda:'')
    app.add_url_rule('/explore','explore',lambda:'')
    app.add_url_rule('/admin/<slug>/thumbnail','admin_publication_thumbnail_preview',lambda slug:'')
    with app.test_request_context('/'):
        return render_template('share.html',lesson=with_publication_scripts(lesson()),
            question='Cube',slug='solve-a-2x2-rubiks-cube',published=True,editorial_preview=True,
            publication={'title':'Cube','summary':'Cube','preview':False},
            audio_urls=[f'/audio/{i}.mp3' for i in range(16)],cube_animation_data=data)

def test_actual_private_template_renders_sixteen_animated_controls_and_saved_audio():
    page=_render(module().private_cube_animation('solve-a-2x2-rubiks-cube',lesson()))
    assert page.count('<canvas ')==16
    assert page.count('data-cube-action="replay"')==16
    assert 'cube-player.js' in page and 'cube-animation-data' in page
    assert 'Slide 16 of 16' in page and '/audio/15.mp3' in page

def test_actual_template_leaves_unbound_lesson_player_unchanged():
    page=_render(None)
    assert '<canvas ' not in page and 'cube-player.js' not in page
    assert 'Slide 16 of 16' in page and '/audio/15.mp3' in page
