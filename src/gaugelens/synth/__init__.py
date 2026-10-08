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
    "DialLabels",
    "DialSpec",
    "RenderedDial",
    "render_dial",
    "render_random",
    "sample_spec",
    "sample_value",
]
