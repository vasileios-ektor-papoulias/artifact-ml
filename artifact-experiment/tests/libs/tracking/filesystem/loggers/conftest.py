import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytest
from artifact_experiment._impl.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment._impl.filesystem.loggers.array_collections import (
    FilesystemArrayCollectionLogger,
)
from artifact_experiment._impl.filesystem.loggers.arrays import FilesystemArrayLogger
from artifact_experiment._impl.filesystem.loggers.plot_collections import (
    FilesystemPlotCollectionLogger,
)
from artifact_experiment._impl.filesystem.loggers.plots import FilesystemPlotLogger
from artifact_experiment._impl.filesystem.loggers.score_collections import (
    FilesystemScoreCollectionLogger,
)
from artifact_experiment._impl.filesystem.loggers.scores import FilesystemScoreLogger
from artifact_experiment._impl.filesystem.native_run import FilesystemRun
from pytest_mock import MockerFixture


@pytest.fixture
def in_memory_df_store(mocker: MockerFixture) -> Dict[str, pd.DataFrame]:
    score_store = {}

    def fake_path_exists(self):
        return str(self) in score_store

    def fake_read_csv(path: Union[Path, str]):
        path_str = str(path)
        if path_str not in score_store:
            raise FileNotFoundError(f"{path_str} not found in store.")
        return score_store[path_str].copy()

    def fake_to_csv(self, path: Union[Path, str], index: bool = True):
        _ = index
        score_store[str(path)] = self.copy()

    mocker.patch("os.makedirs")
    mocker.patch("pandas.DataFrame.to_csv", new=fake_to_csv)
    mocker.patch("pathlib.Path.exists", new=fake_path_exists)
    mocker.patch("pandas.read_csv", side_effect=fake_read_csv)
    return score_store


@pytest.fixture
def expected_logs(request) -> Dict[str, List[float]]:
    logs = {}
    for name, ls_values in request.param.items():
        logs[name] = [request.getfixturevalue(v) for v in ls_values]
    return logs


@pytest.fixture
def mock_incremental_path_generator(mocker: MockerFixture) -> List[str]:
    generated_paths: List[str] = []

    def fake_generate(dir_path: str, fmt: Optional[str] = None) -> str:
        count = 1 + sum(1 for p in generated_paths if p.startswith(dir_path + os.sep))
        path = (
            os.path.join(dir_path, f"{count}.{fmt}")
            if fmt is not None
            else os.path.join(dir_path, str(count))
        )
        generated_paths.append(path)
        return path

    mocker.patch(
        "artifact_experiment.libs.utils.incremental_path_generator.IncrementalPathGenerator.generate",
        side_effect=fake_generate,
    )

    return generated_paths


@pytest.fixture
def score_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemScoreLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemScoreLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemScoreLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemArrayLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemArrayLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemArrayLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemPlotLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemPlotLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemPlotLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemScoreCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemScoreCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemScoreCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemArrayCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemArrayCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemArrayCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemPlotCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemPlotCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemPlotCollectionLogger(run=adapter)
        return adapter, logger

    return _factory
