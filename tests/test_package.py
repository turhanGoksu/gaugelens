from importlib.metadata import version

import gaugelens


def test_version_matches_installed_metadata() -> None:
    assert gaugelens.__version__ == version("gaugelens")
