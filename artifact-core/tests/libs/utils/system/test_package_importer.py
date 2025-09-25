import os
import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import pytest
from artifact_core.libs.utils.system.package_importer import PackageImporter
from pytest_mock import MockerFixture


@pytest.fixture
def mock_sys_path() -> Generator[List[str], None, None]:
    original_path = sys.path.copy()
    yield sys.path
    sys.path.clear()
    sys.path.extend(original_path)


@pytest.fixture
def mock_package_structure() -> List[Tuple[str, str, bool]]:
    return [
        ("", "module1", False),
        ("", "module2", False),
        ("", "subpackage", True),
        ("", "subpackage.submodule1", False),
        ("", "subpackage.submodule2", False),
    ]


@pytest.fixture
def mock_walk_packages(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("pkgutil.walk_packages")  # Important: patch at correct scope


@pytest.fixture
def mock_import_module(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("importlib.import_module")


def _adjust_path_to_os(path: Union[str, Path]) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _add_prefix_to_package_structure(
    prefix: str, package_structure: List[Tuple[str, str, bool]]
) -> List[Tuple[str, str, bool]]:
    return [(importer, prefix + name, ispkg) for importer, name, ispkg in package_structure]


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, root, expected_package_prefix, expected_parent_in_syspath",
    [
        ("/path/to/artifacts", None, "artifacts.", "/path/to"),
        (Path("/path/to/artifacts"), None, "artifacts.", "/path/to"),
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
    ],
)
def test_import_all_from_package_path(
    mock_sys_path: List[str],
    mock_package_structure: List[Tuple[str, str, bool]],
    mock_walk_packages: MagicMock,
    mock_import_module: MagicMock,
    path: Union[str, Path],
    root: Optional[str],
    expected_package_prefix: str,
    expected_parent_in_syspath: str,
):
    os_adjusted_path = _adjust_path_to_os(path)
    os_adjusted_parent = _adjust_path_to_os(expected_parent_in_syspath)
    prefixed_mock_structure = _add_prefix_to_package_structure(
        prefix=expected_package_prefix,
        package_structure=mock_package_structure,
    )
    mock_walk_packages.return_value = prefixed_mock_structure
    PackageImporter.import_all_from_package_path(path=path, root=root)
    assert os_adjusted_parent in map(os.path.normcase, sys.path)
    mock_walk_packages.assert_called_once_with([os_adjusted_path], prefix=expected_package_prefix)
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
