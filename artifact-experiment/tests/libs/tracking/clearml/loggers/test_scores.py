# import os
# from typing import Callable, List, Optional, Tuple

# import pytest
# from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
# from artifact_experiment.libs.tracking.neptune.loggers.scores import NeptuneScoreLogger
# from pytest_mock import MockerFixture


# @pytest.mark.parametrize(
#     "experiment_id, run_id, ls_score_names, ls_scores",
#     [
#         ("exp1", "run1", [], []),
#         ("exp1", "run1", ["score_1"], ["score_1"]),
#         ("exp1", "run1", ["score_1", "score_2"], ["score_1", "score_2"]),
#         ("exp1", "run1", ["score_1", "score_3"], ["score_1", "score_3"]),
#         ("exp1", "run1", ["score_1", "score_2", "score_3"], ["score_1", "score_2", "score_3"]),
#         (
#             "exp1",
#             "run1",
#             ["score_1", "score_2", "score_3", "score_4", "score_5"],
#             ["score_1", "score_2", "score_3", "score_4", "score_5"],
#         ),
#     ],
#     indirect=["ls_scores"],
# )
# def test_log(
#     mocker: MockerFixture,
#     score_logger_factory: Callable[
#         [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneScoreLogger]
#     ],
#     experiment_id: str,
#     run_id: str,
#     ls_score_names: List[str],
#     ls_scores: List[float],
# ):
#     adapter, logger = score_logger_factory(experiment_id, run_id)
#     adapter.log = mocker.MagicMock()
#     for score_name, score in zip(ls_score_names, ls_scores):
#         logger.log(artifact_name=score_name, artifact=score)
#     assert adapter.log.call_count == len(ls_scores)
#     for idx, call_args in enumerate(adapter.log.call_args_list):
#         score_name = ls_score_names[idx]
#         score = ls_scores[idx]
#         expected_path = os.path.join("artifacts", "scores", score_name)
#         assert call_args.kwargs == {"artifact_path": expected_path, "artifact": score}
