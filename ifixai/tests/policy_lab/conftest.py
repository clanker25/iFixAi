import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "examples" / "business_policy_lab.py"
_SPEC = importlib.util.spec_from_file_location("business_lab_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


@pytest.fixture
def native_example():
    return example
