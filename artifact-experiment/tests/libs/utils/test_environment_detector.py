from typing import Callable, Dict, Generator
from unittest.mock import MagicMock

import pytest
from artifact_experiment.libs.utils.environment_detector import Environment, EnvironmentDetector
from pytest_mock import MockerFixture


@pytest.fixture
def terminal_state_constructor(
    mocker: MockerFixture,
) -> Generator[Callable[[], Dict[str, MagicMock]], None, None]:
    def _state_constructor() -> Dict[str, MagicMock]:
        pytest.importorskip("IPython.core.getipython")
        mock_ipython = mocker.patch(
            "artifact_experiment.libs.utils.environment_detector.EnvironmentDetector._get_ipython_shell",
            return_value=None,
        )
        mock_isatty = mocker.patch("sys.stdin.isatty", return_value=True)
        return {"mock_ipython": mock_ipython, "mock_isatty": mock_isatty}

    yield _state_constructor


@pytest.fixture
def ipython_terminal_state_constructor(
    mocker: MockerFixture,
) -> Generator[Callable[[], Dict[str, MagicMock]], None, None]:
    def _state_constructor() -> Dict[str, MagicMock]:
        class DummyTerminalShell:
            pass

        DummyTerminalShell.__module__ = "IPython.terminal.interactiveshell"
        DummyTerminalShell.__name__ = "TerminalInteractiveShell"

        mock_ipython = mocker.patch(
            "artifact_experiment.libs.utils.environment_detector.EnvironmentDetector._get_ipython_shell",
            return_value=DummyTerminalShell(),
        )
        mock_isatty = mocker.patch("sys.stdin.isatty", return_value=False)
        return {"mock_ipython": mock_ipython, "mock_isatty": mock_isatty}

    yield _state_constructor


@pytest.fixture
def jupyter_state_constructor(
    mocker: MockerFixture,
) -> Generator[Callable[[], Dict[str, MagicMock]], None, None]:
    def _state_constructor() -> Dict[str, MagicMock]:
        class DummyZMQShell:
            pass

        DummyZMQShell.__module__ = "ipykernel.zmqshell"
        DummyZMQShell.__name__ = "ZMQInteractiveShell"

        mock_ipython = mocker.patch(
            "artifact_experiment.libs.utils.environment_detector.EnvironmentDetector._get_ipython_shell",
            return_value=DummyZMQShell(),
        )
        mock_isatty = mocker.patch("sys.stdin.isatty", return_value=False)
        return {"mock_ipython": mock_ipython, "mock_isatty": mock_isatty}

    yield _state_constructor


@pytest.fixture
def headless_state_constructor(
    mocker: MockerFixture,
) -> Generator[Callable[[], Dict[str, MagicMock]], None, None]:
    def _state_constructor() -> Dict[str, MagicMock]:
        pytest.importorskip("IPython.core.getipython")
        mock_ipython = mocker.patch(
            "artifact_experiment.libs.utils.environment_detector.EnvironmentDetector._get_ipython_shell",
            return_value=None,
        )
        mock_isatty = mocker.patch("sys.stdin.isatty", return_value=False)
        return {"mock_ipython": mock_ipython, "mock_isatty": mock_isatty}

    yield _state_constructor


@pytest.fixture
def environment_scenario_dispatcher(request) -> Callable[[], Dict[str, MagicMock]]:
    return request.getfixturevalue(f"{request.param}_state_constructor")


@pytest.mark.parametrize(
    "environment_scenario_dispatcher, expected_environment, should_raise",
    [
        ("terminal", Environment.TERMINAL, False),
        ("ipython_terminal", Environment.IPYTHON_TERMINAL, False),
        ("jupyter", Environment.JUPYTER_NOTEBOOK, False),
        ("headless", None, True),
    ],
    indirect=["environment_scenario_dispatcher"],
)
def test_detect(
    environment_scenario_dispatcher: Callable[[], Dict[str, MagicMock]],
    expected_environment: Environment,
    should_raise: bool,
):
    _ = environment_scenario_dispatcher()

    if should_raise:
        with pytest.raises(RuntimeError):
            EnvironmentDetector.detect()
    else:
        result = EnvironmentDetector.detect()
        assert result == expected_environment


@pytest.mark.parametrize(
    "environment_scenario_dispatcher, expected_in_headed",
    [
        ("terminal", True),
        ("ipython_terminal", True),
        ("jupyter", True),
        ("headless", False),
    ],
    indirect=["environment_scenario_dispatcher"],
)
def test_in_headed_environment(
    environment_scenario_dispatcher: Callable[[], Dict[str, MagicMock]],
    expected_in_headed: bool,
):
    _ = environment_scenario_dispatcher()
    result = EnvironmentDetector.in_headed_environment()
    assert result == expected_in_headed
