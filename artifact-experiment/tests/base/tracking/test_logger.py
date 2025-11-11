import os
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_core._base.artifact_dependencies import ArtifactResult
from pytest_mock import MockerFixture

from tests.base.tracking.dummy.adapter import DummyRunAdapter
from tests.base.tracking.dummy.logger import DummyArtifactLogger


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_artifact_names, ls_artifact_results",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["score_1"], ["score_1"]),
        ("exp1", "run1", ["score_1", "score_2"], ["score_1", "score_2"]),
        ("exp1", "run1", ["score_1", "score_3"], ["score_1", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3"], ["score_1", "score_2", "score_3"]),
        (
            "exp1",
            "run1",
            ["score_1", "score_2", "score_3", "score_4", "score_5"],
            ["score_1", "score_2", "score_3", "score_4", "score_5"],
        ),
        (
            "exp1",
            "run1",
            ["score_1", "array_2", "plot_1", "plot_collection_4"],
            ["score_1", "array_2", "plot_1", "plot_collection_4"],
        ),
    ],
    indirect=["ls_artifact_results"],
)
def test_log(
    mocker: MockerFixture,
    logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyRunAdapter, DummyArtifactLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_artifact_names: List[str],
    ls_artifact_results: List[ArtifactResult],
):
    adapter, logger = logger_factory(experiment_id, run_id)
    spy_adapter_log = mocker.spy(adapter, "log")
    for artifact_name, artifact in zip(ls_artifact_names, ls_artifact_results):
        logger.log(artifact_name=artifact_name, artifact=artifact)
    assert spy_adapter_log.call_count == len(ls_artifact_results)
    for idx, call_args in enumerate(spy_adapter_log.call_args_list):
        artifact_name = ls_artifact_names[idx]
        artifact = ls_artifact_results[idx]
        expected_path = os.path.join("", artifact_name)
        assert call_args.kwargs == {"artifact_path": expected_path, "artifact": artifact}
