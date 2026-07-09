from __future__ import annotations

from pathlib import Path

import pytest

_SAMPLES_ROOT = Path(__file__).resolve().parent.parent / "samples" / "tiny_example"


@pytest.fixture
def tiny_index_dir() -> Path:
    """Índice de 5 documentos de `inverted-index-builder` (formato plano),
    reutilizado tal cual del fixture `tiny_example` de `query-parser-autocomplete`."""
    return _SAMPLES_ROOT / "index"
