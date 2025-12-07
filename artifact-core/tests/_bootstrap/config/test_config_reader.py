import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from unittest.mock import MagicMock

import pytest
from artifact_core._bootstrap.config.config_reader import ToolkitConfigReader
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
from artifact_core._utils.collections.map_merger import MapMerger
from pytest import FixtureRequest
from pytest_mock import MockerFixture


@dataclass
class ConfigScenario:
    base_config: Mapping[str, Any]
    override_config: Optional[Mapping[str, Any]]
    base_filepath: Path
    override_dir: Optional[Path]
    expected_result: Mapping[str, Any]


@pytest.fixture
def cfg_simple_base_only(tmp_path: Path) -> ConfigScenario:
    base_config = {"key1": "value1", "key2": "value2"}
    base_filepath = tmp_path / "base" / "config.json"
    base_filepath.parent.mkdir(parents=True)
    base_filepath.write_text(json.dumps(base_config))
    return ConfigScenario(
        base_config=base_config,
        override_config=None,
        base_filepath=base_filepath,
        override_dir=None,
        expected_result=base_config,
    )


@pytest.fixture
def cfg_with_override(tmp_path: Path) -> ConfigScenario:
    base_config = {"key1": "value1", "nested": {"a": 1}}
    base_filepath = tmp_path / "base" / "config.json"
    base_filepath.parent.mkdir(parents=True)
    base_filepath.write_text(json.dumps(base_config))
    override_config = {"key2": "value2", "nested": {"b": 2}}
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    override_filepath = override_dir / "toolkit.json"
    override_filepath.write_text(json.dumps(override_config))
    expected_result = {
        "key1": "value1",
        "key2": "value2",
        "nested": {"a": 1, "b": 2},
    }
    return ConfigScenario(
        base_config=base_config,
        override_config=override_config,
        base_filepath=base_filepath,
        override_dir=override_dir,
        expected_result=expected_result,
    )


@pytest.fixture
def cfg_override_dir_no_file(tmp_path: Path) -> ConfigScenario:
    base_config = {"key1": "value1"}
    base_filepath = tmp_path / "base" / "config.json"
    base_filepath.parent.mkdir(parents=True)
    base_filepath.write_text(json.dumps(base_config))
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    return ConfigScenario(
        base_config=base_config,
        override_config=None,
        base_filepath=base_filepath,
        override_dir=override_dir,
        expected_result=base_config,
    )


@pytest.fixture
def cfg_nested_merge(tmp_path: Path) -> ConfigScenario:
    base_config = {
        "scores": {"metric1": {"param": "value1"}},
        "arrays": {"arr1": {"size": 10}},
    }
    base_filepath = tmp_path / "base" / "config.json"
    base_filepath.parent.mkdir(parents=True)
    base_filepath.write_text(json.dumps(base_config))
    override_config = {
        "scores": {
            "metric1": {"param": "override"},
            "metric2": {"new": "value"},
        },
        "plots": {"plot1": {"type": "line"}},
    }
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    override_filepath = override_dir / "toolkit.json"
    override_filepath.write_text(json.dumps(override_config))
    expected_result = {
        "scores": {
            "metric1": {"param": "override"},
            "metric2": {"new": "value"},
        },
        "arrays": {"arr1": {"size": 10}},
        "plots": {"plot1": {"type": "line"}},
    }

    return ConfigScenario(
        base_config=base_config,
        override_config=override_config,
        base_filepath=base_filepath,
        override_dir=override_dir,
        expected_result=expected_result,
    )


@pytest.fixture
def cfg_scenario_dispatcher(
    request: FixtureRequest,
) -> Callable[[str], ConfigScenario]:
    def _get_scenario(scenario_name: str) -> ConfigScenario:
        fixture_name = f"cfg_{scenario_name}"
        return request.getfixturevalue(fixture_name)

    return _get_scenario


@pytest.fixture
def mock_toolkit() -> MagicMock:
    toolkit = MagicMock(spec=DomainToolkit)
    toolkit.config_override_filename = "toolkit.json"
    return toolkit


@pytest.mark.unit
@pytest.mark.parametrize(
    "scenario_name",
    [
        "simple_base_only",
        "with_override",
        "override_dir_no_file",
        "nested_merge",
    ],
)
def test_read(
    mocker: MockerFixture,
    mock_toolkit: MagicMock,
    cfg_scenario_dispatcher: Callable[[str], ConfigScenario],
    scenario_name: str,
):
    scenario = cfg_scenario_dispatcher(scenario_name)
    mock_toolkit.base_config_filepath = scenario.base_filepath
    spy_json_load = mocker.spy(json, "load")
    spy_open = mocker.spy(Path, "open")
    spy_merge = mocker.spy(MapMerger, "merge")
    result = ToolkitConfigReader.read(
        domain_toolkit=mock_toolkit, override_dir=scenario.override_dir
    )
    has_override = scenario.override_config is not None
    expected_loads = 2 if has_override else 1
    assert spy_json_load.call_count == expected_loads
    spy_open.assert_any_call(scenario.base_filepath, "r", encoding="utf-8")
    if has_override:
        override_filepath = scenario.override_dir / mock_toolkit.config_override_filename
        spy_open.assert_any_call(override_filepath, "r", encoding="utf-8")
        assert spy_merge.call_count > 0
    else:
        spy_merge.assert_not_called()
    assert result == scenario.expected_result
