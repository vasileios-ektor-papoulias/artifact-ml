import os
from pathlib import Path
from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.client import FilesystemTrackingClient
from artifact_experiment.libs.tracking.filesystem.loggers.array_collections import (
    FilesystemArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.arrays import FilesystemArrayLogger
from artifact_experiment.libs.tracking.filesystem.loggers.plot_collections import (
    FilesystemPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.plots import FilesystemPlotLogger
from artifact_experiment.libs.tracking.filesystem.loggers.score_collections import (
    FilesystemScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.scores import FilesystemScoreLogger
from artifact_experiment.libs.tracking.filesystem.native_run import FilesystemRun
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.arrays import NeptuneArrayLogger
from artifact_experiment.libs.tracking.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.plots import NeptunePlotLogger
from artifact_experiment.libs.tracking.neptune.loggers.score_collections import (
    NeptuneScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.scores import NeptuneScoreLogger
from pytest_mock import MockerFixture


@pytest.fixture
def native_run_factory(
    mocker: MockerFixture,
) -> Callable[[Optional[str], Optional[str]], FilesystemRun]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> FilesystemRun:
        if experiment_id is None:
            experiment_id = "default_experiment_id"
        if run_id is None:
            run_id = "default_run_id"
        mocker.patch.object(FilesystemRun, "_root_dir", new=Path("test_root"))
        mocker.patch("artifact_experiment.libs.tracking.filesystem.native_run.os.makedirs")
        mocker.patch(
            "artifact_experiment.libs.tracking.filesystem.native_run.DirectoryOpenButton",
            autospec=True,
        )
        mocker.patch("artifact_experiment.libs.tracking.filesystem.native_run.print")
        native_run = FilesystemRun(experiment_id=experiment_id, run_id=run_id)
        return native_run

    return _factory


@pytest.fixture
def patch_filesystem_run_creation(mocker: MockerFixture):
    mocker.patch.object(FilesystemRun, "_root_dir", new=Path("test_root"))
    mocker.patch("artifact_experiment.libs.tracking.filesystem.native_run.os.makedirs")
    mocker.patch(
        "artifact_experiment.libs.tracking.filesystem.native_run.DirectoryOpenButton", autospec=True
    )
    mocker.patch("artifact_experiment.libs.tracking.filesystem.native_run.print")


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], FilesystemRun],
) -> Callable[[Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRun, FilesystemRunAdapter]:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"

        native_run = native_run_factory(experiment_id, run_id)
        adapter = FilesystemRunAdapter(native_run=native_run)
        return native_run, adapter

    return _factory


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, NeptuneRunAdapter],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
    ]:
        _, adapter = adapter_factory(experiment_id, run_id)
        score_logger = NeptuneScoreLogger(run=adapter)
        array_logger = NeptuneArrayLogger(run=adapter)
        plot_logger = NeptunePlotLogger(run=adapter)
        score_collection_logger = NeptuneScoreCollectionLogger(run=adapter)
        array_collection_logger = NeptuneArrayCollectionLogger(run=adapter)
        plot_collection_logger = NeptunePlotCollectionLogger(run=adapter)
        return (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
        )

    return _factory


@pytest.fixture
def client_factory(
    loggers_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            FilesystemRunAdapter,
            FilesystemScoreLogger,
            FilesystemArrayLogger,
            FilesystemPlotLogger,
            FilesystemScoreCollectionLogger,
            FilesystemArrayCollectionLogger,
            FilesystemPlotCollectionLogger,
        ],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        FilesystemRunAdapter,
        FilesystemScoreLogger,
        FilesystemArrayLogger,
        FilesystemPlotLogger,
        FilesystemScoreCollectionLogger,
        FilesystemArrayCollectionLogger,
        FilesystemPlotCollectionLogger,
        FilesystemTrackingClient,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[
        FilesystemRunAdapter,
        FilesystemScoreLogger,
        FilesystemArrayLogger,
        FilesystemPlotLogger,
        FilesystemScoreCollectionLogger,
        FilesystemArrayCollectionLogger,
        FilesystemPlotCollectionLogger,
        FilesystemTrackingClient,
    ]:
        (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
        ) = loggers_factory(experiment_id, run_id)
        client = FilesystemTrackingClient(
            run=adapter,
            score_logger=score_logger,
            array_logger=array_logger,
            plot_logger=plot_logger,
            score_collection_logger=score_collection_logger,
            array_collection_logger=array_collection_logger,
            plot_collection_logger=plot_collection_logger,
        )
        return (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
            client,
        )

    return _factory


@pytest.fixture
def get_absolute_log_path() -> Callable[[str, str, str], str]:
    def _prepend(experiment_id: str, run_id: str, path_relative_to_run_dir: str) -> str:
        return os.path.join(
            "test_dir", "artifact_ml", experiment_id, run_id, path_relative_to_run_dir.lstrip("/")
        ).replace("\\", "/")

    return _prepend
