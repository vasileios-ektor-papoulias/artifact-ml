# import os
# from typing import Callable, Dict, List, Optional, Tuple

# import pytest
# from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
# from artifact_experiment.libs.tracking.neptune.loggers.score_collections import (
#     NeptuneScoreCollectionLogger,
# )
# from pytest_mock import MockerFixture


# @pytest.mark.unit
@pytest.mark.parametrize(
#     "experiment_id, run_id, ls_score_collection_names, ls_score_collections",
#     [
#         ("exp1", "run1", [], []),
#         ("exp1", "run1", ["score_collection_1"], ["score_collection_1"]),
#         (
#             "exp1",
#             "run1",
#             ["score_collection_1", "score_collection_2"],
#             ["score_collection_1", "score_collection_2"],
#         ),
#         (
#             "exp1",
#             "run1",
#             ["score_collection_1", "score_collection_3"],
#             ["score_collection_1", "score_collection_3"],
#         ),
#         (
#             "exp1",
#             "run1",
#             ["score_collection_1", "score_collection_2", "score_collection_3"],
#             ["score_collection_1", "score_collection_2", "score_collection_3"],
#         ),
#         (
#             "exp1",
#             "run1",
#             [
#                 "score_collection_1",
#                 "score_collection_2",
#                 "score_collection_3",
#                 "score_collection_4",
#                 "score_collection_5",
#             ],
#             [
#                 "score_collection_1",
#                 "score_collection_2",
#                 "score_collection_3",
#                 "score_collection_4",
#                 "score_collection_5",
#             ],
#         ),
#     ],
#     indirect=["ls_score_collections"],
# )
# def test_log(
#     mocker: MockerFixture,
#     score_collection_logger_factory: Callable[
#         [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneScoreCollectionLogger]
#     ],
#     experiment_id: str,
#     run_id: str,
#     ls_score_collection_names: List[str],
#     ls_score_collections: List[Dict[str, float]],
# ):
#     adapter, logger = score_collection_logger_factory(experiment_id, run_id)
#     adapter.log = mocker.MagicMock()
#     for score_collection_name, score_collection in zip(
#         ls_score_collection_names, ls_score_collections
#     ):
#         logger.log(artifact_name=score_collection_name, artifact=score_collection)
#     assert adapter.log.call_count == len(ls_score_collections)
#     for idx, call_args in enumerate(adapter.log.call_args_list):
#         score_collection_name = ls_score_collection_names[idx]
#         score_collection = ls_score_collections[idx]
#         expected_path = os.path.join("artifacts", "score_collections", score_collection_name)
#         assert call_args.kwargs == {"artifact_path": expected_path, "artifact": score_collection}
