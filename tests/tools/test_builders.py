from __future__ import annotations

from pathlib import Path

import pytest

from tools.gallery import build_gallery
from tools.memo_builder import build_memo


def test_memo_builder_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_memo([], output=Path("dummy.md"))


def test_gallery_builder_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_gallery(Path("run_dir"), output=Path("gallery"))
