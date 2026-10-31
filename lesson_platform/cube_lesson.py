"""Bind the authored private cube animations only to their exact clean scripts."""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from .library import publication_audio_text

SLUG = 'solve-a-2x2-rubiks-cube'

@lru_cache(maxsize=1)
def _authored_data():
    path = Path(__file__).with_name('cube_animation_v2.json')
    data = json.loads(path.read_bytes())
    if data.get('version') != 2 or data.get('slug') != SLUG:
        raise ValueError('Unexpected cube animation version')
    if len(data.get('script_hashes', [])) != 16 or len(data.get('scenes', [])) != 16:
        raise ValueError('Incomplete cube animation')
    if any(not isinstance(h,str) or not re.fullmatch('[a-f0-9]{64}',h) for h in data['script_hashes']):
        raise ValueError('Invalid script binding')
    for scene in data['scenes']:
        stickers, moves = scene.get('initial'), scene.get('moves')
        if not isinstance(stickers,list) or len(stickers)!=24 or not isinstance(moves,list) or len(moves)>160:
            raise ValueError('Incomplete scene')
        seen, colors = set(), {}
        for s in stickers:
            p,n,c=s.get('position'),s.get('normal'),s.get('color')
            if (not isinstance(p,list) or len(p)!=3 or any(type(v)is not int or v not in (-1,1) for v in p)
                or not isinstance(n,list) or len(n)!=3 or any(type(v)is not int or v not in (-1,0,1) for v in n)
                or sum(abs(v) for v in n)!=1 or any(v and p[i]!=v for i,v in enumerate(n)) or c not in {'W','Y','G','B','R','O'}):
                raise ValueError('Invalid cube sticker')
            seen.add((tuple(p),tuple(n)));colors[c]=colors.get(c,0)+1
        if len(seen)!=24 or len(colors)!=6 or any(v!=4 for v in colors.values()):
            raise ValueError('Invalid cube state')
        if any(not isinstance(m,str) or not re.fullmatch(r"[RLUDFBxyz](?:'|2)?",m) for m in moves):
            raise ValueError('Invalid move')
        if any(not isinstance(scene.get(k,''),str) or len(scene.get(k,''))>400 for k in ('title','caption')):
            raise ValueError('Invalid scene copy')
    return data

def private_cube_animation(slug, lesson):
    """Old text, another topic, or incomplete source receives no animation override."""
    if slug != SLUG or not isinstance(lesson,dict) or len(lesson.get('slides',[]))!=16:
        return None
    try:
        data = _authored_data()
        hashes = [hashlib.sha256(publication_audio_text(s).encode()).hexdigest() for s in lesson['slides']]
        if hashes != data['script_hashes']:
            return None
        return copy.deepcopy(data)
    except (OSError, ValueError, TypeError, KeyError):
        return None
