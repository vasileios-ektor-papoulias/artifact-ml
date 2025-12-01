import os
from pathlib import Path
from typing import Union

import pytest
from artifact_core._utils.filesystem.separator_replacer import SeparatorReplacer


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, new_separator, expected_result",
    [
        (f"path{os.sep}to{os.sep}file", ".", "path.to.file"),
        (f"path{os.sep}to{os.sep}file", "_", "path_to_file"),
        (f"path{os.sep}to{os.sep}file", "-", "path-to-file"),
        (f"path{os.sep}to{os.sep}file", "::", "path::to::file"),
        (Path(f"path{os.sep}to{os.sep}file"), ".", "path.to.file"),
        ("simple", ".", "simple"),
        ("", ".", ""),
        (f"path{os.sep}with.dots{os.sep}file", "_", "path_with.dots_file"),
        (f"a{os.sep}b{os.sep}c", ".", "a.b.c"),
    ],
)
def test_replace_separator(path: Union[Path, str], new_separator: str, expected_result: str):
    result = SeparatorReplacer.replace_separator(path=path, new=new_separator)
    assert isinstance(result, str)
    assert result == expected_result
