from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Optional

import pytest
from artifact_core._utils.filesystem.directory_locator import DirectoryLocator
from pytest import FixtureRequest


@dataclass
class DirectoryScenario:
    expected_marker: Optional[Path]
    start_dir: Path


@pytest.fixture
def dir_marker_at_root(tmp_path: Path) -> Generator[DirectoryScenario, None, None]:
    marker_dir = tmp_path / ".marker"
    marker_dir.mkdir()
    yield DirectoryScenario(expected_marker=marker_dir, start_dir=tmp_path)


@pytest.fixture
def dir_marker_in_parent(tmp_path: Path) -> Generator[DirectoryScenario, None, None]:
    marker_dir = tmp_path / ".marker"
    marker_dir.mkdir()
    child = tmp_path / "child"
    child.mkdir()
    yield DirectoryScenario(expected_marker=marker_dir, start_dir=child)


@pytest.fixture
def dir_marker_in_grandparent(tmp_path: Path) -> Generator[DirectoryScenario, None, None]:
    marker_dir = tmp_path / ".marker"
    marker_dir.mkdir()
    child = tmp_path / "child"
    child.mkdir()
    grandchild = child / "grandchild"
    grandchild.mkdir()
    yield DirectoryScenario(expected_marker=marker_dir, start_dir=grandchild)


@pytest.fixture
def dir_marker_not_found(tmp_path: Path) -> Generator[DirectoryScenario, None, None]:
    start = tmp_path / "subdir"
    start.mkdir()
    yield DirectoryScenario(expected_marker=None, start_dir=start)


@pytest.fixture
def dir_closest_marker(tmp_path: Path) -> Generator[DirectoryScenario, None, None]:
    root_marker = tmp_path / ".marker"
    root_marker.mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_marker = nested / ".marker"
    nested_marker.mkdir()
    deep = nested / "deep"
    deep.mkdir()
    yield DirectoryScenario(expected_marker=nested_marker, start_dir=deep)


@pytest.fixture
def dir_marker_is_file(tmp_path: Path) -> Generator[DirectoryScenario, None, None]:
    marker_file = tmp_path / ".marker"
    marker_file.touch()
    yield DirectoryScenario(expected_marker=None, start_dir=tmp_path)


@pytest.fixture
def dir_scenario_dispatcher(request: FixtureRequest) -> Callable[[str], DirectoryScenario]:
    def _get_scenario(scenario_name: str) -> DirectoryScenario:
        fixture_name = f"dir_{scenario_name}"
        return request.getfixturevalue(fixture_name)

    return _get_scenario


@pytest.mark.unit
@pytest.mark.parametrize(
    "scenario_name, marker_name, expected_found",
    [
        ("marker_at_root", ".marker", True),
        ("marker_in_parent", ".marker", True),
        ("marker_in_grandparent", ".marker", True),
        ("marker_not_found", ".marker", False),
        ("closest_marker", ".marker", True),
        ("marker_is_file", ".marker", False),
    ],
)
def test_find(
    dir_scenario_dispatcher: Callable[[str], DirectoryScenario],
    scenario_name: str,
    marker_name: str,
    expected_found: bool,
):
    scenario = dir_scenario_dispatcher(scenario_name)
    result = DirectoryLocator.find(marker=marker_name, start=scenario.start_dir)
    if expected_found:
        assert result is not None
        assert result == scenario.expected_marker
        assert result.exists()
        assert result.is_dir()
    else:
        assert result is None
