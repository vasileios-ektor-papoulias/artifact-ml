from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock

import pytest
from artifact_core._bootstrap.config.override_locator import ConfigOverrideLocator
from pytest_mock import MockerFixture


@pytest.fixture
def mock_directory_locator(
    mocker: MockerFixture,
) -> Callable[[Optional[Path]], MagicMock]:
    def _mock_find(return_value: Optional[Path]) -> MagicMock:
        patch_path = "artifact_core._utils.filesystem.directory_locator.DirectoryLocator.find"
        return mocker.patch(patch_path, return_value=return_value)

    return _mock_find


@pytest.mark.unit
@pytest.mark.parametrize(
    "mock_return_value",
    [Path("/home/user/project/.artifact-ml"), Path("/tmp/.artifact-ml"), None],
)
def test_find(
    mock_directory_locator: Callable[[Optional[Path]], MagicMock],
    mock_return_value: Optional[Path],
):
    mock_find = mock_directory_locator(mock_return_value)
    result = ConfigOverrideLocator.find()
    mock_find.assert_called_once_with(marker=".artifact-ml")
    assert result == mock_return_value
