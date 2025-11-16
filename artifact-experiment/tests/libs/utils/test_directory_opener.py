import subprocess
import sys
from typing import Any, Tuple

import pytest
from artifact_experiment._utils.directory_opener import DirectoryOpener
from pytest_mock import MockerFixture


@pytest.fixture
def platform(mocker, request) -> str:
    platform = request.param
    mocker.patch.object(sys, "platform", platform)
    return platform


@pytest.mark.unit
@pytest.mark.parametrize(
    "platform, expected_popen_args",
    [
        ("win32", ('start "" "C:\\test\\path"',)),
        ("darwin", (["open", "/test/path"],)),
        ("linux", (["xdg-open", "/test/path"],)),
    ],
    indirect=["platform"],
)
def test_open_directory(mocker: MockerFixture, platform: str, expected_popen_args: Tuple[Any]):
    mock_popen = mocker.patch("subprocess.Popen")
    path = "C:\\test\\path" if platform.startswith("win") else "/test/path"
    DirectoryOpener.open_directory(path=path)
    mock_popen.assert_called_once()
    called_args, called_kwargs = mock_popen.call_args
    assert called_args == expected_popen_args
    assert called_kwargs["stdout"] == subprocess.DEVNULL
    assert called_kwargs["stderr"] == subprocess.DEVNULL


@pytest.mark.unit
@pytest.mark.parametrize(
    "platform, path, exists, expected_msg",
    [
        ("win32", "C:\\test\\path", True, "Directory is available at: C:\\test\\path"),
        ("darwin", "/test/path", True, "Directory is available at: /test/path"),
        ("linux", "/test/path", True, "Directory is available at: /test/path"),
        ("unknownOS", "/unknown/path", True, "Directory is available at: /unknown/path"),
        ("win32", "C:\\test\\path", False, "Directory does not exist: C:\\test\\path"),
        ("darwin", "/test/path", False, "Directory does not exist: /test/path"),
        ("linux", "/test/path", False, "Directory does not exist: /test/path"),
        ("unknownOS", "/unknown/path", False, "Directory does not exist: /unknown/path"),
    ],
    indirect=["platform"],
)
def test_open_directory_fallback_output(
    mocker: MockerFixture,
    capfd: pytest.CaptureFixture,
    platform: str,
    path: str,
    exists: bool,
    expected_msg: str,
):
    mocker.patch("subprocess.Popen", side_effect=OSError("Failure"))
    mocker.patch("os.path.isdir", return_value=exists)
    _ = platform
    DirectoryOpener.open_directory(path=path)
    out, _ = capfd.readouterr()
    assert expected_msg in out
