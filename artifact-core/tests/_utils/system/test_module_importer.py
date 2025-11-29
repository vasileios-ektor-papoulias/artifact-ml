import os
import sys
from pathlib import Path
from typing import Callable, Generator, List, Tuple, Union
from unittest.mock import MagicMock

import pytest
from artifact_core._utils.system.module_importer import ModuleImporter
from pytest_mock import MockerFixture


@pytest.fixture
def mock_sys_path() -> Generator[List[str], None, None]:
    original_path = sys.path.copy()
    yield sys.path
    sys.path.clear()
    sys.path.extend(original_path)


@pytest.fixture
def mock_package_structure_factory() -> Callable[[str], List[Tuple[str, str, bool]]]:
    def _create_structure(prefix: str = "") -> List[Tuple[str, str, bool]]:
        base_structure = [
            ("", "module1", False),
            ("", "module2", False),
            ("", "subpackage", True),
            ("", "subpackage.submodule1", False),
            ("", "subpackage.submodule2", False),
        ]
        if prefix:
            return [(importer, prefix + name, ispkg) for importer, name, ispkg in base_structure]
        return base_structure

    return _create_structure


@pytest.fixture
def mock_walk_packages(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("pkgutil.walk_packages")


@pytest.fixture
def mock_import_module(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("importlib.import_module")


def _adjust_path_to_os(path: Union[str, Path]) -> str:
    return os.path.normcase(str(Path(path).resolve()))


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, root, expected_package_prefix, expected_parent_in_syspath",
    [
        (
            "/path/to/artifact_core/table_comparison/artifacts",
            "/path/to/artifact_core",
            "artifact_core.table_comparison.artifacts.",
            "/path/to",
        ),
        (
            Path("/path/to/artifact_core/table_comparison/artifacts"),
            "/path/to/artifact_core",
            "artifact_core.table_comparison.artifacts.",
            "/path/to",
        ),
        (
            Path("/path/to/artifact_core/table_comparison/artifacts"),
            Path("/path/to/artifact_core"),
            "artifact_core.table_comparison.artifacts.",
            "/path/to",
        ),
        (
            "/path/to/my_package/subdir/module",
            "/path/to/my_package",
            "my_package.subdir.module.",
            "/path/to",
        ),
    ],
)
def test_import_all_from_package_path(
    mock_sys_path: List[str],
    mock_package_structure_factory: Callable[[str], List[Tuple[str, str, bool]]],
    mock_walk_packages: MagicMock,
    mock_import_module: MagicMock,
    path: Union[str, Path],
    root: Union[str, Path],
    expected_package_prefix: str,
    expected_parent_in_syspath: str,
):
    os_adjusted_path = _adjust_path_to_os(path)
    os_adjusted_parent = _adjust_path_to_os(expected_parent_in_syspath)
    prefixed_mock_structure = mock_package_structure_factory(expected_package_prefix)
    mock_walk_packages.return_value = prefixed_mock_structure
    ModuleImporter.import_modules(path=path, root=root)
    assert os_adjusted_parent in map(os.path.normcase, sys.path)
    actual_call_args = mock_walk_packages.call_args
    actual_path = os.path.normcase(actual_call_args[0][0][0])
    actual_prefix = actual_call_args[1]["prefix"]
    assert actual_path == os_adjusted_path
    assert actual_prefix == expected_package_prefix
    assert mock_walk_packages.call_count == 1
    expected_imports = [
        expected_package_prefix + "module1",
        expected_package_prefix + "module2",
        expected_package_prefix + "subpackage",
        expected_package_prefix + "subpackage.submodule1",
        expected_package_prefix + "subpackage.submodule2",
    ]
    assert mock_import_module.call_count == len(expected_imports)
    for expected_import in expected_imports:
        mock_import_module.assert_any_call(name=expected_import)
