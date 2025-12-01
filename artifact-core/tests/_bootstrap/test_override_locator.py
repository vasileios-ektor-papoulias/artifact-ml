from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from artifact_core._bootstrap.config.override_locator import ConfigOverrideLocator


@pytest.mark.unit
@pytest.mark.parametrize(
    "mock_return_value",
    [
        Path("/home/user/project/.artifact-ml"),
        Path("/tmp/.artifact-ml"),
        None,
    ],
)
def test_find(mock_return_value: Optional[Path]):
    with patch(
        "artifact_core._bootstrap.config.override_locator.DirectoryLocator.find"
    ) as mock_find:
        mock_find.return_value = mock_return_value
        result = ConfigOverrideLocator.find()
        mock_find.assert_called_once_with(marker=".artifact-ml")
        assert result == mock_return_value
