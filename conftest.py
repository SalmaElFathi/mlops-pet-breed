"""Skip the data-validation tests when the DVC-tracked data is not on disk (e.g. in CI)."""

from pathlib import Path

import pytest

DATA_TESTS = {"test_manifest.py"}


def pytest_collection_modifyitems(config, items):
    if Path("data/manifest.json").exists():
        return
    skip = pytest.mark.skip(reason="data/manifest.json is DVC-tracked and not pulled")
    for item in items:
        if item.path.name in DATA_TESTS:
            item.add_marker(skip)
