import pytest
from artifact_experiment.libs.ui.directory_open_button import DirectoryOpenButton
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, description",
    [
        ("/some/path", "Open Directory"),
        ("C:\\windows\\path", "Open Windows Folder"),
    ],
)
def test_button(mocker: MockerFixture, path: str, description: str):
    mock_dir_opener = mocker.patch(
        "artifact_experiment.libs.ui.directory_open_button.DirectoryOpener"
    )
    btn = DirectoryOpenButton(path=path, description=description)
    assert btn.button.description == description
    btn.click()
    mock_dir_opener.open_directory.assert_called_once_with(path=path)
