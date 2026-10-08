"""Synthetic gauge generator (needs the ``[synth]`` extra).

Ground truth is exact by construction: the needle is drawn at the angle that
:mod:`gaugelens.geometry` assigns to the sampled value.
"""

try:
    import PIL  # noqa: F401
except ImportError as error:
    raise ImportError(
        "gaugelens.synth needs Pillow: pip install 'gaugelens[synth]'"
    ) from error

from gaugelens.synth.backgrounds import BackgroundPool, download_polyhaven
from gaugelens.synth.capture import (
    CaptureParams,
    Scene,
    SceneLabels,
    capture,
    render_scene,
    sample_capture,
)
from gaugelens.synth.dial import (
    DialLabels,
    DialSpec,
    RenderedDial,
    render_dial,
    render_random,
    sample_spec,
    sample_value,
)

__all__ = [
    "BackgroundPool",
    "CaptureParams",
    "DialLabels",
    "DialSpec",
    "RenderedDial",
    "Scene",
    "SceneLabels",
    "capture",
    "download_polyhaven",
    "render_dial",
    "render_random",
    "render_scene",
    "sample_capture",
    "sample_spec",
    "sample_value",
]
