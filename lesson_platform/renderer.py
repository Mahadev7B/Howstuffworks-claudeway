"""Render drawing JSON specs to PNG/GIF bytes.

Declarative & safe: we only read known fields from the spec. Unknown shape
types and unknown properties are ignored. No exec(), no eval(), no dynamic
imports — the spec is pure data, never executed.
"""
import copy
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; must be set before importing pyplot

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

logger = logging.getLogger(__name__)

# Keep animations short on Render's modest CPU
ANIM_DURATION_S = 1.6
ANIM_FPS = 12


def _make_fig(spec: dict):
    width = int(spec.get("width", 640))
    height = int(spec.get("height", 240))
    bg = spec.get("background", "#FDF8F3")
    fig, ax = plt.subplots(figsize=(width / 80, height / 80), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # top-left origin like SVG
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    return fig, ax, bg


def _build_patches(ax, shapes):
    """Return [(shape, artist|None)] for each shape. Unknown types produce None."""
    artists = []
    for s in shapes:
        t = s.get("type")
        fill = s.get("fill", "#999999")
        stroke = s.get("stroke", "none")
        sw = s.get("stroke_width", 0)
        artist = None
        try:
            if t == "rect":
                artist = mpatches.Rectangle(
                    (s["x"], s["y"]), s["w"], s["h"],
                    facecolor=fill, edgecolor=stroke, linewidth=sw,
                )
                ax.add_patch(artist)
            elif t == "circle":
                artist = mpatches.Circle(
                    (s["cx"], s["cy"]), s["r"],
                    facecolor=fill, edgecolor=stroke, linewidth=sw,
                )
                ax.add_patch(artist)
            elif t == "ellipse":
                artist = mpatches.Ellipse(
                    (s["cx"], s["cy"]), s["rx"] * 2, s["ry"] * 2,
                    facecolor=fill, edgecolor=stroke, linewidth=sw,
                )
                ax.add_patch(artist)
            elif t == "polygon":
                artist = mpatches.Polygon(
                    s["points"], closed=True,
                    facecolor=fill, edgecolor=stroke, linewidth=sw,
                )
                ax.add_patch(artist)
            elif t == "line":
                (artist,) = ax.plot(
                    [s["x1"], s["x2"]], [s["y1"], s["y2"]],
                    color=s.get("stroke", "#333"),
                    linewidth=s.get("stroke_width", 1.5),
                    linestyle=s.get("dash", "-"),
                )
            elif t == "text":
                artist = ax.text(
                    s["x"], s["y"], str(s.get("text", "")),
                    color=s.get("fill", "#1F1B2E"),
                    fontsize=s.get("size", 12),
                    fontweight=s.get("weight", "normal"),
                    ha=s.get("anchor", "left"), va="center",
                    family="sans-serif",
                )
            elif t == "label":
                ax.annotate(
                    str(s.get("text", "")),
                    xy=s["pointTo"], xytext=(s["x"], s["y"]),
                    color=s.get("fill", "#534AB7"),
                    fontsize=s.get("size", 12),
                    fontweight="bold", ha="center", va="center",
                    family="sans-serif",
                    arrowprops=dict(
                        arrowstyle="-", linestyle="--",
                        color=s.get("fill", "#534AB7"), lw=1,
                    ),
                )
                # Labels are not animatable; no artist handle stored.
            # unknown types silently skipped
        except Exception:
            logger.warning("Skipping malformed shape: %r", t)
        artists.append((s, artist))
    return artists


def _apply_translate(artist, shape: dict, dx: float, dy: float) -> None:
    t = shape.get("type")
    if t == "rect":
        artist.set_x(shape["x"] + dx)
        artist.set_y(shape["y"] + dy)
    elif t in ("circle", "ellipse"):
        artist.center = (shape["cx"] + dx, shape["cy"] + dy)
    elif t == "polygon":
        pts = [(p[0] + dx, p[1] + dy) for p in shape["points"]]
        artist.set_xy(pts)


def _has_animations(spec: dict) -> bool:
    return any(isinstance(s, dict) and "animate" in s for s in spec.get("shapes", []))


def render_png(spec: dict) -> bytes:
    fig, ax, bg = _make_fig(spec)
    _build_patches(ax, spec.get("shapes", []))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=bg, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def render_gif(spec: dict, duration_s: float = ANIM_DURATION_S,
               fps: int = ANIM_FPS) -> bytes:
    fig, ax, _bg = _make_fig(spec)
    originals = copy.deepcopy(spec.get("shapes", []))
    artists = _build_patches(ax, originals)
    anim_items = [(s, a) for s, a in artists if a is not None and "animate" in s]
    total = max(int(duration_s * fps), 2)

    def update(frame: int):
        t = frame / max(total - 1, 1)
        for shape, artist in anim_items:
            anim = shape.get("animate") or {}
            prog = (1 - abs(1 - 2 * t)) if anim.get("yoyo", True) else t
            dx = float(anim.get("dx", 0)) * prog
            dy = float(anim.get("dy", 0)) * prog
            try:
                _apply_translate(artist, shape, dx, dy)
            except Exception:
                pass
        return [a for _, a in anim_items if a is not None]

    ani = FuncAnimation(fig, update, frames=total, interval=1000 / fps, blit=False)

    # PillowWriter wants a file path; write to temp file then read bytes
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        ani.save(str(path), writer=PillowWriter(fps=fps))
        plt.close(fig)
        return path.read_bytes()
    finally:
        try:
            path.unlink()
        except Exception:
            pass


def render(spec: dict) -> tuple[bytes, str]:
    """Choose PNG or GIF based on whether spec has any animations.

    Returns (image_bytes, mime_type).
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dict")
    if _has_animations(spec):
        return render_gif(spec), "image/gif"
    return render_png(spec), "image/png"
