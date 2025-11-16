import json
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import pytest
from artifact_core._bootstrap.libs.config_identifier import DomainToolkit, ToolkitConfigIdentifier
from artifact_core._bootstrap.libs.config_merger import ConfigMerger
from artifact_core._bootstrap.libs.override_locator import ConfigOverrideLocator
from pytest_mock import MockerFixture


@pytest.fixture
def temp_dir_with_config(
    tmp_path: Path,
) -> Generator[Tuple[Path, Path, Path, Dict[str, Any]], None, None]:
    config_dirpath = tmp_path / ".artifact-ml"
    config_dirpath.mkdir()
    config_filepath = config_dirpath / ToolkitConfigIdentifier.get_config_filename(
        domain_toolkit=DomainToolkit.TABLE_COMPARISON
    )
    config_data = {"test": "value", "nested": {"key": "value"}}
    with open(config_filepath, "w") as f:
        json.dump(config_data, f)
    subdir = tmp_path / "subdir" / "deeper"
    subdir.mkdir(parents=True)
    yield tmp_path, config_dirpath, config_filepath, config_data
    if config_filepath.exists():
        config_filepath.unlink()
    if config_dirpath.exists():
        config_dirpath.rmdir()
    if (tmp_path / "subdir" / "deeper").exists():
        (tmp_path / "subdir" / "deeper").rmdir()
    if (tmp_path / "subdir").exists():
        (tmp_path / "subdir").rmdir()


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_exists, expected_result",
    [
        (True, True),
        (False, False),
    ],
)
def test_get_config_override(
    temp_dir_with_config: Tuple[Path, Path, Path, Dict[str, Any]],
    mocker: MockerFixture,
    config_exists: bool,
    expected_result: bool,
):
    tmp_path, _, config_filepath, config_data = temp_dir_with_config
    if not config_exists and config_filepath.exists():
        config_filepath.unlink()
    mocker.patch.object(Path, "cwd", return_value=tmp_path)
    result = ConfigOverrideLocator.get_config_override(DomainToolkit.TABLE_COMPARISON)
    if expected_result:
        assert result == config_data
    else:
        assert result is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_config, override, expected",
    [
        ({"key1": "value1", "key2": "value2"}, None, {"key1": "value1", "key2": "value2"}),
        ({"key1": "value1"}, {"key2": "value2"}, {"key1": "value1", "key2": "value2"}),
        (
            {"key1": "value1", "key2": "value2"},
            {"key1": "new_value"},
            {"key1": "new_value", "key2": "value2"},
        ),
        (
            {"key1": {"nested1": "value1"}},
            {"key2": "value2"},
            {"key1": {"nested1": "value1"}, "key2": "value2"},
        ),
        (
            {"key1": {"nested1": "value1", "nested2": "value2"}},
            {"key1": {"nested1": "new_value", "nested3": "value3"}},
            {"key1": {"nested1": "new_value", "nested2": "value2", "nested3": "value3"}},
        ),
        (
            {"key1": {"nested1": {"deep1": "value1"}}},
            {"key1": {"nested1": {"deep2": "value2"}}},
            {"key1": {"nested1": {"deep1": "value1", "deep2": "value2"}}},
        ),
        ({"key1": {"nested1": "value1"}}, {"key1": "new_value"}, {"key1": "new_value"}),
        ({}, {"key1": "value1"}, {"key1": "value1"}),
        ({"key1": "value1"}, {}, {"key1": "value1"}),
    ],
)
def test_merge(
    base_config: Dict[str, Any], override: Optional[Dict[str, Any]], expected: Dict[str, Any]
):
    result = ConfigMerger.merge(base_config, override)
    assert result == expected
