"""Preview: render sample drawing JSON specs locally using the production renderer.

Run:
  pip install matplotlib pillow
  python preview_render.py

Writes preview_*.png and preview_*.gif in the current directory.
"""
import copy
from pathlib import Path

from lesson_platform.renderer import render_gif, render_png

# --- Sample specs ---------------------------------------------------------

ROCKET = {
    "width": 640, "height": 240,
    "background": "#B5D4F4",
    "shapes": [
        {"type": "ellipse", "cx": 80,  "cy": 50,  "rx": 40, "ry": 18, "fill": "#FFFFFF"},
        {"type": "ellipse", "cx": 560, "cy": 50,  "rx": 42, "ry": 20, "fill": "#FFFFFF"},
        {"type": "ellipse", "cx": 520, "cy": 180, "rx": 35, "ry": 15, "fill": "#FFFFFF"},
        {"type": "ellipse", "cx": 100, "cy": 175, "rx": 32, "ry": 14, "fill": "#FFFFFF"},
        {"type": "rect",    "x": 305, "y": 75, "w": 30, "h": 105, "fill": "#7F77DD"},
        {"type": "rect",    "x": 310, "y": 100, "w": 20, "h": 38, "fill": "#EEEDFE"},
        {"type": "circle",  "cx": 320, "cy": 119, "r": 7, "fill": "#185FA5"},
        {"type": "circle",  "cx": 320, "cy": 119, "r": 3, "fill": "#FFFFFF"},
        {"type": "polygon", "points": [[320, 18], [340, 75], [300, 75]], "fill": "#534AB7"},
        {"type": "rect",    "x": 302, "y": 145, "w": 36, "h": 8, "fill": "#AFA9EC"},
        {"type": "polygon", "points": [[305, 155], [280, 190], [305, 190]], "fill": "#534AB7"},
        {"type": "polygon", "points": [[335, 155], [360, 190], [335, 190]], "fill": "#534AB7"},
        {"type": "polygon", "points": [[308, 180], [332, 180], [320, 215]], "fill": "#FAC775"},
        {"type": "polygon", "points": [[312, 188], [328, 188], [320, 210]], "fill": "#EF9F27"},
        {"type": "label", "text": "nose cone", "x": 460, "y": 30,  "pointTo": [330, 30],  "fill": "#534AB7", "size": 13},
        {"type": "label", "text": "body",      "x": 460, "y": 120, "pointTo": [340, 120], "fill": "#534AB7", "size": 13},
        {"type": "label", "text": "fins",      "x": 200, "y": 160, "pointTo": [285, 180], "fill": "#534AB7", "size": 13},
        {"type": "label", "text": "flame",     "x": 200, "y": 210, "pointTo": [312, 205], "fill": "#EF9F27", "size": 13},
    ],
}

ROCKET_ANIMATED = copy.deepcopy(ROCKET)
ROCKET_ANIMATED["shapes"][0]["animate"] = {"dx": 20, "dy": 0, "yoyo": True}
ROCKET_ANIMATED["shapes"][1]["animate"] = {"dx": -20, "dy": 0, "yoyo": True}
for i in range(4, 12):
    ROCKET_ANIMATED["shapes"][i]["animate"] = {"dx": 0, "dy": -6, "yoyo": True}
ROCKET_ANIMATED["shapes"][12]["animate"] = {"dx": 0, "dy": 6, "yoyo": True}
ROCKET_ANIMATED["shapes"][13]["animate"] = {"dx": 0, "dy": 4, "yoyo": True}


def _write(path: str, data: bytes) -> None:
    Path(path).write_bytes(data)
    print(f"Wrote {path} ({len(data)//1024} KB)")


if __name__ == "__main__":
    _write("preview_rocket.png", render_png(ROCKET))
    _write("preview_rocket.gif", render_gif(ROCKET_ANIMATED))
    print("\nOpen the PNGs/GIFs to judge quality.")
