import sys
from pathlib import Path

import pytest

# Make `starter_kit`, `solver` and `period_router` importable: they all live in
# the Computational Track folder, two levels above this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starter_kit import build_hardware_graph  # noqa: E402

from period_router.chip import make_chip  # noqa: E402


@pytest.fixture
def hw():
    """The challenge's 20-qubit hardware graph."""
    return build_hardware_graph()


@pytest.fixture
def chip(hw):
    return make_chip(hw)
