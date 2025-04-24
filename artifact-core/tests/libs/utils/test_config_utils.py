import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from artifact_core.libs.utils.config_utils import (
    ConfigMerger,
    ConfigOverrideLocator,
    EngineConfigType,
)


@pytest.fixture
def temp_dir_with_config(tmp_path):
    config_path = tmp_path / ".validation_engine"
    config_path.mkdir()
    config_file = config_path / EngineConfigType.TABLE_COMPARISON.value
    config_data = {"test": "value", "nested": {"key": "value"}}
    with open(config_file, "w") as f:
        json.dump(config_data, f)
    subdir = tmp_path / "subdir" / "deeper"
    subdir.mkdir(parents=True)
    yield tmp_path, config_path, config_file, config_data
    if config_file.exists():
        config_file.unlink()
    if config_path.exists():
        config_path.rmdir()
    if (tmp_path / "subdir" / "deeper").exists():
        (tmp_path / "subdir" / "deeper").rmdir()
    if (tmp_path / "subdir").exists():
        (tmp_path / "subdir").rmdir()


@pytest.mark.parametrize(
    "config_exists, expected_result",
    [
        (True, True),
        (False, False),
    ],
)
def test_get_config_override(
    temp_dir_with_config, monkeypatch, config_exists: bool, expected_result: bool
):
    tmp_path, _, config_file, config_data = temp_dir_with_config

    if not config_exists and config_file.exists():
        config_file.unlink()
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    result = ConfigOverrideLocator.get_config_override(EngineConfigType.TABLE_COMPARISON)
    if expected_result:
        assert result == config_data
    else:
        assert result is None


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
