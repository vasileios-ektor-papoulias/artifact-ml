from pathlib import Path
from typing import Optional, Union

import pytest
from artifact_core._utils.filesystem.path_normalizer import PathResolver


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, expected_is_absolute, expected_is_none",
    [
        (".", True, False),
        ("..", True, False),
        ("relative/path", True, False),
        (Path("relative/path"), True, False),
        (Path("."), True, False),
        ("", True, False),
        (None, False, True),
    ],
)
def test_resolve(
    path: Optional[Union[Path, str]],
    expected_is_absolute: bool,
    expected_is_none: bool,
):
    result = PathResolver.resolve(path=path)
    if expected_is_none:
        assert result is None
    else:
        assert result is not None
        assert isinstance(result, str)
        if expected_is_absolute:
            assert Path(result).is_absolute()


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        ".",
        "..",
        "relative/path",
        Path("relative/path"),
        "some/nested/directory/file.txt",
    ],
)
def test_resolve_idempotent(path: Union[Path, str]):
    result1 = PathResolver.resolve(path=path)
    assert result1 is not None
    result2 = PathResolver.resolve(path=result1)
    assert result2 is not None
    assert result1 == result2
