"""gaugelens: read analog dial gauges from photos.

Every reading comes with an explicit status; the library never returns a
number without one.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gaugelens")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
