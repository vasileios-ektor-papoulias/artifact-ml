import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Type
from unittest.mock import MagicMock

import pytest
from artifact_core._bootstrap import toolkit_initializer
from artifact_core._bootstrap.config.toolkit_config import ToolkitConfig
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
from artifact_core._bootstrap.toolkit_initializer import ToolkitInitializer
from pytest import FixtureRequest
from pytest_mock import MockerFixture


@dataclass
class InitScenario:
    override_dir: Optional[Path]
    custom_artifacts_dir: Optional[Path]
    expected_import_calls: int


@pytest.fixture
def init_no_override() -> InitScenario:
    return InitScenario(
        override_dir=None,
        custom_artifacts_dir=None,
        expected_import_calls=1,
    )


@pytest.fixture
def init_override_no_custom() -> InitScenario:
    return InitScenario(
        override_dir=Path("/home/user/.artifact-ml"),
        custom_artifacts_dir=None,
        expected_import_calls=1,
    )


@pytest.fixture
def init_override_with_custom() -> InitScenario:
    return InitScenario(
        override_dir=Path("/home/user/.artifact-ml"),
        custom_artifacts_dir=Path("custom/artifacts"),
        expected_import_calls=2,
    )


@pytest.fixture
def init_no_override_with_custom() -> InitScenario:
    return InitScenario(
        override_dir=None,
        custom_artifacts_dir=Path("custom/artifacts"),
        expected_import_calls=1,
    )


@pytest.fixture
def init_scenario_dispatcher(
    request: FixtureRequest,
) -> Callable[[str], InitScenario]:
    def _get_scenario(scenario_name: str) -> InitScenario:
        fixture_name = f"init_{scenario_name}"
        return request.getfixturevalue(fixture_name)

    return _get_scenario


@pytest.fixture
def mock_toolkit() -> MagicMock:
    toolkit = MagicMock(spec=DomainToolkit)
    toolkit.native_artifacts_dir = Path("/pkg/toolkit/_artifacts")
    toolkit.package_root = Path("/pkg")
    return toolkit


@pytest.fixture
def mock_toolkit_config_factory() -> Callable[[Optional[Path]], ToolkitConfig]:
    def _create_config(custom_artifacts_dir: Optional[Path] = None) -> ToolkitConfig:
        return ToolkitConfig(
            custom_artifacts_dir=custom_artifacts_dir,
            scores_config={},
            arrays_config={},
            plots_config={},
            score_collections_config={},
            array_collections_config={},
            plot_collections_config={},
        )

    return _create_config


@pytest.fixture(autouse=True)
def fresh_toolkit_initializer() -> Type[ToolkitInitializer]:
    importlib.reload(toolkit_initializer)
    from artifact_core._bootstrap.toolkit_initializer import (
        ToolkitInitializer as FreshToolkitInitializer,
    )

    return FreshToolkitInitializer


@pytest.mark.unit
@pytest.mark.parametrize(
    "override_dir_value",
    [Path("/home/user/.artifact-ml"), Path("/tmp/.artifact-ml"), None],
)
def test_load_toolkit_config(
    fresh_toolkit_initializer: Type[ToolkitInitializer],
    mocker: MockerFixture,
    mock_toolkit: MagicMock,
    mock_toolkit_config_factory: Callable[[Optional[Path]], ToolkitConfig],
    override_dir_value: Optional[Path],
):
    expected_config = mock_toolkit_config_factory(None)
    mock_find = mocker.patch(
        "artifact_core._bootstrap.config.override_locator.ConfigOverrideLocator.find",
        return_value=override_dir_value,
    )
    mock_load = mocker.patch(
        "artifact_core._bootstrap.config.toolkit_config.ToolkitConfig.load",
        return_value=expected_config,
    )
    result = fresh_toolkit_initializer.load_toolkit_config(domain_toolkit=mock_toolkit)
    mock_find.assert_called_once()
    mock_load.assert_called_once_with(
        domain_toolkit=mock_toolkit, config_override_dir=override_dir_value
    )
    assert result == expected_config


@pytest.mark.unit
@pytest.mark.parametrize(
    "scenario_name",
    ["no_override", "override_no_custom", "override_with_custom", "no_override_with_custom"],
)
def test_init_toolkit(
    fresh_toolkit_initializer: Type[ToolkitInitializer],
    mocker: MockerFixture,
    mock_toolkit: MagicMock,
    mock_toolkit_config_factory: Callable[[Optional[Path]], ToolkitConfig],
    init_scenario_dispatcher: Callable[[str], InitScenario],
    scenario_name: str,
):
    scenario = init_scenario_dispatcher(scenario_name)
    mock_config = mock_toolkit_config_factory(scenario.custom_artifacts_dir)
    mocker.patch(
        "artifact_core._bootstrap.config.override_locator.ConfigOverrideLocator.find",
        return_value=scenario.override_dir,
    )
    mock_load = mocker.patch(
        "artifact_core._bootstrap.config.toolkit_config.ToolkitConfig.load",
        return_value=mock_config,
    )
    mock_import = mocker.patch(
        "artifact_core._utils.system.module_importer.ModuleImporter.import_modules"
    )
    fresh_toolkit_initializer.init_toolkit(domain_toolkit=mock_toolkit)
    mock_load.assert_called_once_with(
        domain_toolkit=mock_toolkit, config_override_dir=scenario.override_dir
    )
    assert mock_import.call_count == scenario.expected_import_calls
    mock_import.assert_any_call(
        path=mock_toolkit.native_artifacts_dir, root=mock_toolkit.package_root
    )
    if scenario.expected_import_calls == 2:
        assert scenario.override_dir is not None
        assert scenario.custom_artifacts_dir is not None
        mock_import.assert_any_call(
            path=scenario.custom_artifacts_dir,
            root=scenario.override_dir.parent,
        )
