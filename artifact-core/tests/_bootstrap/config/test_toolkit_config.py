from pathlib import Path
from typing import Any, Mapping, Optional
from unittest.mock import MagicMock

import pytest
from artifact_core._bootstrap.config.toolkit_config import ToolkitConfig
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
from pytest_mock import MockerFixture


@pytest.fixture
def mock_toolkit() -> MagicMock:
    return MagicMock(spec=DomainToolkit)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_data, override_dir, expected_result",
    [
        (
            {"scores": {"metric1": {"param": "value"}}, "arrays": {"arr1": {}}},
            None,
            ToolkitConfig(
                custom_artifacts_dir=None,
                scores_config={"metric1": {"param": "value"}},
                arrays_config={"arr1": {}},
                plots_config={},
                score_collections_config={},
                array_collections_config={},
                plot_collections_config={},
            ),
        ),
        (
            {"custom_artifacts_dir": "custom/artifacts", "scores": {}},
            None,
            ToolkitConfig(
                custom_artifacts_dir=Path("custom/artifacts"),
                scores_config={},
                arrays_config={},
                plots_config={},
                score_collections_config={},
                array_collections_config={},
                plot_collections_config={},
            ),
        ),
        (
            {},
            None,
            ToolkitConfig(
                custom_artifacts_dir=None,
                scores_config={},
                arrays_config={},
                plots_config={},
                score_collections_config={},
                array_collections_config={},
                plot_collections_config={},
            ),
        ),
        (
            {"scores": {"m1": {}}, "plots": {"p1": {}}},
            Path("/override"),
            ToolkitConfig(
                custom_artifacts_dir=None,
                scores_config={"m1": {}},
                arrays_config={},
                plots_config={"p1": {}},
                score_collections_config={},
                array_collections_config={},
                plot_collections_config={},
            ),
        ),
    ],
)
def test_load(
    mocker: MockerFixture,
    mock_toolkit: MagicMock,
    config_data: Mapping[str, Any],
    override_dir: Optional[Path],
    expected_result: ToolkitConfig,
):
    mock_read = mocker.patch(
        "artifact_core._bootstrap.config.toolkit_config.ToolkitConfigReader.read",
        return_value=config_data,
    )
    result = ToolkitConfig.load(domain_toolkit=mock_toolkit, config_override_dir=override_dir)
    mock_read.assert_called_once_with(domain_toolkit=mock_toolkit, override_dir=override_dir)
    assert result == expected_result
