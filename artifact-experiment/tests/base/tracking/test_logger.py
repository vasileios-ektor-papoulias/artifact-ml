from typing import Callable, List, Optional, Tuple

import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from pytest_mock import MockerFixture

from tests.base.tracking.dummy.adapter import DummyRunAdapter
from tests.base.tracking.dummy.logger import DummyArtifactLogger


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_artifact_results",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_1"]),
        ("exp1", "run1", ["score_1", "score_2"]),
        ("exp1", "run1", ["score_1", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3", "score_4", "score_5"]),
        ("exp1", "run1", ["score_1", "array_2", "plot_1", "plot_collection_4"]),
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
    ls_artifact_results: List[ArtifactResult],
):
    adapter, logger = logger_factory(experiment_id, run_id)
    print(type(adapter))
    adapter.log = mocker.MagicMock()
    for idx, artifact in enumerate(ls_artifact_results):
        logger.log(artifact_name="artifact", artifact=artifact)
    assert adapter.log.call_count == len(ls_artifact_results)
    for idx, call_args in enumerate(adapter.log.call_args_list):
        artifact = ls_artifact_results[idx]
        assert call_args.kwargs == {"artifact_path": "", "artifact": artifact}
