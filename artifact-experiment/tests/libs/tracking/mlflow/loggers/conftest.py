import os
from typing import Callable, Dict, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment._impl.mlflow.adapter import MlflowNativeRun, MlflowRunAdapter
from artifact_experiment._impl.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment._impl.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment._impl.mlflow.loggers.plot_collections import (
    MlflowPlotCollectionLogger,
)
from artifact_experiment._impl.mlflow.loggers.plots import MlflowPlotLogger
from artifact_experiment._impl.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from artifact_experiment._impl.mlflow.loggers.scores import MlflowScoreLogger


@pytest.fixture
def mock_tempdir(mocker) -> Dict[str, MagicMock]:
    fake_temp_dir_path = os.path.join("mock", "tmp", "dir")
    mock_tempdir_cm = mocker.patch("tempfile.TemporaryDirectory")
    tempdir_instance = mock_tempdir_cm.return_value
    tempdir_instance.__enter__.return_value = fake_temp_dir_path
    tempdir_instance.name = fake_temp_dir_path
    return {"mock_tempdir_cm": mock_tempdir_cm}


@pytest.fixture
def score_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowScoreLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MlflowRunAdapter, MlflowScoreLogger]:
        _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
        logger = MlflowScoreLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowArrayLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MlflowRunAdapter, MlflowArrayLogger]:
        _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
        logger = MlflowArrayLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowPlotLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MlflowRunAdapter, MlflowPlotLogger]:
        _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
        logger = MlflowPlotLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowScoreCollectionLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MlflowRunAdapter, MlflowScoreCollectionLogger]:
        _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
        logger = MlflowScoreCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowArrayCollectionLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MlflowRunAdapter, MlflowArrayCollectionLogger]:
        _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
        logger = MlflowArrayCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowPlotCollectionLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MlflowRunAdapter, MlflowPlotCollectionLogger]:
        _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
        logger = MlflowPlotCollectionLogger(run=adapter)
        return adapter, logger

    return _factory
