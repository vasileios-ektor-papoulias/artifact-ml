import os
from typing import Callable, Dict, Generator, Optional
from unittest.mock import MagicMock

import pytest
from artifact_experiment.libs.utils.environment_variable_reader import (
    EnvironmentVariableReader,
)
from pytest_mock import MockerFixture


@pytest.fixture
def var_set_state_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Callable[[str, str], Dict[str, MagicMock]], None, None]:
    def _state_constructor(name: str, value: str) -> Dict[str, MagicMock]:
        monkeypatch.setenv(name=name, value=value)
        return {}

    yield _state_constructor


@pytest.fixture
def var_not_set_headed_env_state_constructor(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> Generator[Callable[[str, str], Dict[str, MagicMock]], None, None]:
    def _state_constructor(name: str, value: str) -> Dict[str, MagicMock]:
        monkeypatch.delenv(name=name, raising=False)
        mock_in_headed_environemt = mocker.patch(
            "artifact_experiment.libs.utils.environment_detector.EnvironmentDetector.in_headed_environment",
            return_value=True,
        )
        mock_getpass = mocker.patch(
            "artifact_experiment.libs.utils.environment_variable_reader.getpass",
            return_value=value,
        )
        return {
            "mock_in_headed_environemt": mock_in_headed_environemt,
            "mock_getpass": mock_getpass,
        }

    yield _state_constructor


@pytest.fixture
def var_not_set_headless_env_state_constructor(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> Generator[Callable[[str, str], Dict[str, MagicMock]], None, None]:
    def _state_constructor(name: str, value: str) -> Dict[str, MagicMock]:
        _ = value
        monkeypatch.delenv(name, raising=False)
        mock_in_headed_environemt = mocker.patch(
            "artifact_experiment.libs.utils.environment_detector.EnvironmentDetector.in_headed_environment",
            return_value=False,
        )
        return {"mock_in_headed_environemt": mock_in_headed_environemt}

    yield _state_constructor


@pytest.fixture
def environment_scenario_dispatcher(request) -> Callable[[str, str], Dict[str, MagicMock]]:
    return request.getfixturevalue(f"{request.param}_state_constructor")


@pytest.mark.unit
@pytest.mark.parametrize(
    "environment_scenario_dispatcher, env_var_name, env_var_value, set_env, "
    + "expected_var_value, should_raise",
    [
        ("var_set", "MY_ENV_VAR", "env_value", False, "env_value", False),
        (
            "var_not_set_headed_env",
            "MY_ENV_VAR",
            "env_value",
            False,
            "env_value",
            False,
        ),
        (
            "var_not_set_headed_env",
            "MY_ENV_VAR",
            "env_value",
            True,
            "env_value",
            False,
        ),
        ("var_not_set_headless_env", "MY_ENV_VAR", "env_value", False, None, True),
    ],
    indirect=["environment_scenario_dispatcher"],
)
def test_get(
    environment_scenario_dispatcher: Callable[[str, str], Dict[str, MagicMock]],
    env_var_name: str,
    env_var_value: str,
    set_env: bool,
    expected_var_value: Optional[str],
    should_raise: bool,
):
    context = environment_scenario_dispatcher(env_var_name, env_var_value)

    if should_raise:
        with pytest.raises(RuntimeError):
            EnvironmentVariableReader.get(env_var_name=env_var_name)
    else:
        result = EnvironmentVariableReader.get(
            env_var_name,
            set_env=set_env,
        )
        assert result == expected_var_value
        if set_env:
            assert os.environ[env_var_name] == expected_var_value
        if "mock_getpass" in context:
            expected_prompt = "MY_ENV_VAR: "
            context["mock_getpass"].assert_called_once_with(prompt=expected_prompt)
