# from typing import Callable, Optional, Tuple
# from unittest.mock import MagicMock
# from uuid import UUID

# import pytest
# from artifact_core.base.artifact_dependencies import ArtifactResult
# from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter, NeptuneRunStatus


# @pytest.mark.unit
@pytest.mark.parametrize(
#     "experiment_id, run_id",
#     [
#         ("exp1", "run1"),
#         ("exp1", None),
#     ],
# )
# def test_build(
#     patch_neptune_run_creation,
#     experiment_id: str,
#     run_id: Optional[str],
# ):
#     adapter = NeptuneRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
#     assert adapter.is_active is True
#     assert adapter.experiment_id == experiment_id
#     if run_id is not None:
#         assert adapter.run_id == run_id
#     else:
#         assert adapter.run_id is not None
#         assert len(adapter.run_id) == STANDARD_UUID_LENGTH
#         UUID(adapter.run_id)


# @pytest.mark.unit
@pytest.mark.parametrize(
#     "experiment_id, run_id",
#     [
#         ("exp1", "run1"),
#         ("exp1", None),
#     ],
# )
# def test_from_native_run(
#     native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
#     experiment_id: str,
#     run_id: Optional[str],
# ):
#     native_run = native_run_factory(experiment_id, run_id)
#     adapter = NeptuneRunAdapter.from_native_run(native_run=native_run)
#     assert adapter.is_active is True
#     assert adapter.experiment_id == experiment_id
#     assert adapter.experiment_id == native_run.experiment_id
#     assert adapter.run_id == native_run.run_id
#     if run_id is not None:
#         assert adapter.run_id == run_id
#         assert native_run.run_id == run_id


# @pytest.mark.unit
@pytest.mark.parametrize(
#     "experiment_id, run_id",
#     [
#         ("exp1", "run1"),
#         ("exp1", None),
#     ],
# )
# def test_native_context_manager(
#     adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
#     experiment_id: str,
#     run_id: Optional[str],
# ):
#     native_run, adapter = adapter_factory(experiment_id, run_id)
#     with adapter.native() as ctx_native_run:
#         assert ctx_native_run is native_run


# @pytest.mark.unit
@pytest.mark.parametrize(
#     "experiment_id, run_id",
#     [
#         ("exp1", "run1"),
#         ("exp1", None),
#     ],
# )
# def test_stop_run(
#     adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
#     experiment_id: str,
#     run_id: Optional[str],
# ):
#     _, adapter = adapter_factory(experiment_id, run_id)
#     assert adapter.is_active
#     assert adapter.run_status == NeptuneRunStatus.RUNNING
#     adapter.stop()
#     assert not adapter.is_active
#     assert adapter.run_status == NeptuneRunStatus.INACTIVE


# @pytest.mark.unit
@pytest.mark.parametrize(
#     "experiment_id, run_id, artifact_path, artifact_result",
#     [
#         ("exp1", "run1", "/test/path/1", "score_1"),
#         ("exp1", None, "/test/path/1", "score_2"),
#         ("exp1", "run1", "/test/path/1", "array_1"),
#         ("exp1", None, "/test/path/1", "array_2"),
#         ("exp1", "run1", "/test/path/1", "plot_1"),
#         ("exp1", None, "/test/path/1", "plot_2"),
#         ("exp1", "run1", "/test/path/1", "score_collection_1"),
#         ("exp1", None, "/test/path/1", "score_collection_2"),
#         ("exp1", "run1", "/test/path/1", "array_collection_1"),
#         ("exp1", None, "/test/path/1", "array_collection_2"),
#         ("exp1", "run1", "/test/path/1", "plot_collection_1"),
#         ("exp1", None, "/test/path/1", "plot_collection_2"),
#     ],
#     indirect=["artifact_result"],
# )
# def test_log(
#     get_absolute_log_path: Callable[[str], str],
#     adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
#     experiment_id: str,
#     run_id: str,
#     artifact_path: str,
#     artifact_result: ArtifactResult,
# ):
#     native_run, adapter = adapter_factory(experiment_id, run_id)
#     adapter.log(artifact_path=artifact_path, artifact=artifact_result)
#     log_path = get_absolute_log_path(artifact_path)
#     native_run[log_path].append.assert_called_once_with(artifact_result)


# @pytest.mark.unit
@pytest.mark.parametrize(
#     "experiment_id, run_id, path_source, dir_target",
#     [
#         ("exp1", "run1", "/test/path", "uploads"),
#         ("exp1", None, "/data/models/model.pkl", "models"),
#         ("exp1", "run1", "/logs/experiment.log", "logs"),
#         ("exp1", None, "/artifacts/plot.png", "plots"),
#         ("exp1", "run1", "/results/summary.json", "results"),
#     ],
# )
# def test_upload(
#     get_absolute_log_path: Callable[[str], str],
#     adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
#     experiment_id: str,
#     run_id: Optional[str],
#     path_source: str,
#     dir_target: str,
# ):
#     native_run, adapter = adapter_factory(experiment_id, run_id)
#     adapter.upload(path_source=path_source, dir_target=dir_target)

#     expected_path = get_absolute_log_path(dir_target)
#     native_run[expected_path].upload.assert_called_once_with(path_source)
