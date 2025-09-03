import os
import shutil
import tempfile
from typing import Callable, Generator, List

import pytest
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator


@pytest.fixture
def tmp_dir_factory() -> Generator[Callable[[List[str]], str], None, None]:
    ls_created_dirs = []

    def _factory(ls_filenames: List[str]) -> str:
        tmp_dir_path = tempfile.mkdtemp()
        for filename in ls_filenames:
            open(os.path.join(tmp_dir_path, filename), "a").close()
        ls_created_dirs.append(tmp_dir_path)
        return tmp_dir_path

    try:
        yield _factory
    finally:
        for tmp_dir in ls_created_dirs:
            shutil.rmtree(tmp_dir)


@pytest.mark.unit
@pytest.mark.parametrize(
    "existing_files, fmt, expected_filename",
    [
        ([], None, "0"),
        (["0"], None, "1"),
        (["0", "1", "2"], None, "3"),
        (["0", "1", "2"], ".txt", "0.txt"),
        (["0", "1", "2", "3"], ".png", "0.png"),
        (["0", "1", "2", "0.png"], ".png", "1.png"),
        (["0", "1", "2", "4.png"], ".png", "5.png"),
        (["0.txt", "1.txt"], ".txt", "2.txt"),
        (["0.txt", "1.txt", "0.log"], ".txt", "2.txt"),
        (["0.txt", "1.txt", "0.log"], ".log", "1.log"),
        (["0.txt", "1.txt", "3.log"], ".log", "4.log"),
        (["a.txt", "b.txt"], ".txt", "0.txt"),
        (["10", "3", "5"], None, "11"),
    ],
)
def test_generate(tmp_dir_factory, existing_files, fmt, expected_filename):
    tmp_dir = tmp_dir_factory(existing_files)
    result_path = IncrementalPathGenerator.generate(dir_path=tmp_dir, fmt=fmt)
    assert os.path.basename(result_path) == expected_filename


@pytest.mark.unit
@pytest.mark.parametrize(
    "next_idx, fmt, expected_filename",
    [
        (0, None, "0"),
        (1, None, "1"),
        (5, ".csv", "5.csv"),
        (42, ".log", "42.log"),
    ],
)
def test_format_path(tmp_dir_factory, next_idx, fmt, expected_filename):
    tmp_dir = tmp_dir_factory([])
    result_path = IncrementalPathGenerator.format_path(dir_path=tmp_dir, next_idx=next_idx, fmt=fmt)
    assert os.path.basename(result_path) == expected_filename
