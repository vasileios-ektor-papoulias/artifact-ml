from typing import Callable, List, Tuple, Type

import pandas as pd
import pytest
from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
    TabularDataSpec,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArtifact,
    TableComparisonArtifactResources,
)
from artifact_experiment.base.callbacks.tracking import (
    ArrayCallbackHandler,
    ArrayCollectionCallbackHandler,
    PlotCallbackHandler,
    PlotCollectionCallbackHandler,
    ScoreCallbackHandler,
    ScoreCollectionCallbackHandler,
)
from artifact_experiment.table_comparison.callback_factory import TableComparisonCallbackFactory
from artifact_experiment.table_comparison.resources import TableComparisonCallbackResources
from pytest_mock import MockerFixture

from tests.table_comparison.validation_plan.dummy.validation_plan import DummyTableComparisonPlan


def test_build(
    mocker: MockerFixture,
    tabular_data_spec: TabularDataSpec,
    expected_score_types: List[TableComparisonScoreType],
    expected_plot_types: List[TableComparisonPlotType],
    expected_score_collection_types: List[TableComparisonScoreCollectionType],
    expected_array_collection_types: List[TableComparisonArrayCollectionType],
    expected_plot_collection_types: List[TableComparisonPlotCollectionType],
):
    spy_build_score = mocker.spy(TableComparisonCallbackFactory, "build_score_callback")
    spy_build_array = mocker.spy(TableComparisonCallbackFactory, "build_array_callback")
    spy_build_plot = mocker.spy(TableComparisonCallbackFactory, "build_plot_callback")
    spy_build_score_collection = mocker.spy(
        TableComparisonCallbackFactory, "build_score_collection_callback"
    )
    spy_build_array_collection = mocker.spy(
        TableComparisonCallbackFactory, "build_array_collection_callback"
    )
    spy_build_plot_collection = mocker.spy(
        TableComparisonCallbackFactory, "build_plot_collection_callback"
    )
    plan = DummyTableComparisonPlan.build(resource_spec=tabular_data_spec)
    assert isinstance(plan, DummyTableComparisonPlan)
    assert spy_build_score.call_count == len(expected_score_types)
    for score_type in expected_score_types:
        spy_build_score.assert_any_call(score_type=score_type, resource_spec=tabular_data_spec)
    assert spy_build_array.call_count == 0
    assert spy_build_plot.call_count == len(expected_plot_types)
    for plot_type in expected_plot_types:
        spy_build_plot.assert_any_call(plot_type=plot_type, resource_spec=tabular_data_spec)
    assert spy_build_score_collection.call_count == len(expected_score_collection_types)
    for sct in expected_score_collection_types:
        spy_build_score_collection.assert_any_call(
            score_collection_type=sct, resource_spec=tabular_data_spec
        )
    assert spy_build_array_collection.call_count == len(expected_array_collection_types)
    for act in expected_array_collection_types:
        spy_build_array_collection.assert_any_call(
            array_collection_type=act, resource_spec=tabular_data_spec
        )
    assert spy_build_plot_collection.call_count == len(expected_plot_collection_types)
    for pct in expected_plot_collection_types:
        spy_build_plot_collection.assert_any_call(
            plot_collection_type=pct, resource_spec=tabular_data_spec
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset_real_dispatcher, dataset_synthetic_dispatcher",
    [("df_1", "df_2"), ("df_3", "df_4")],
    indirect=["dataset_real_dispatcher", "dataset_synthetic_dispatcher"],
)
def test_execute(
    mocker: MockerFixture,
    expected_score_types: List[TableComparisonScoreType],
    expected_plot_types: List[TableComparisonPlotType],
    expected_score_collection_types: List[TableComparisonScoreCollectionType],
    expected_array_collection_types: List[TableComparisonArrayCollectionType],
    expected_plot_collection_types: List[TableComparisonPlotCollectionType],
    expected_artifact_classes: List[Type[TableComparisonArtifact]],
    dataset_real_dispatcher: pd.DataFrame,
    dataset_synthetic_dispatcher: pd.DataFrame,
):
    ls_artifact_compute_spies = [mocker.spy(cls, "compute") for cls in expected_artifact_classes]
    df_real = dataset_real_dispatcher
    df_synthetic = dataset_synthetic_dispatcher
    data_spec = TabularDataSpec.from_df(df=df_real)
    expected_artifact_resources = TableComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    plan = DummyTableComparisonPlan.build(resource_spec=data_spec)
    assert set(plan.arrays.keys()) == set()
    assert set(plan.array_collections.keys()) == set()
    assert set(plan.scores.keys()) == set()
    assert set(plan.score_collections.keys()) == set()
    assert set(plan.plots.keys()) == set()
    assert set(plan.plot_collections.keys()) == set()
    plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
    for spy in ls_artifact_compute_spies:
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert "resources" in kwargs
        assert kwargs["resources"] == expected_artifact_resources
    assert set(plan.scores.keys()) == set(t.name for t in expected_score_types)
    assert set(plan.plots.keys()) == set(t.name for t in expected_plot_types)
    assert set(plan.score_collections.keys()) == set(
        t.name for t in expected_score_collection_types
    )
    assert set(plan.array_collections.keys()) == set(
        t.name for t in expected_array_collection_types
    )
    assert set(plan.plot_collections.keys()) == set(t.name for t in expected_plot_collection_types)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset_real_dispatcher, dataset_synthetic_dispatcher",
    [("df_1", "df_2"), ("df_3", "df_4")],
    indirect=["dataset_real_dispatcher", "dataset_synthetic_dispatcher"],
)
def test_execute_runs_all_handlers(
    mocker: MockerFixture,
    handlers_factory: Callable[
        [],
        Tuple[
            ScoreCallbackHandler,
            ArrayCallbackHandler,
            PlotCallbackHandler,
            ScoreCollectionCallbackHandler,
            ArrayCollectionCallbackHandler,
            PlotCollectionCallbackHandler,
        ],
    ],
    dataset_real_dispatcher: pd.DataFrame,
    dataset_synthetic_dispatcher: pd.DataFrame,
):
    tup_handlers = handlers_factory()
    plan = DummyTableComparisonPlan(*tup_handlers)
    df_real = dataset_real_dispatcher
    df_synthetic = dataset_synthetic_dispatcher
    expected_callback_resources = TableComparisonCallbackResources.build(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    ls_handler_spies = []
    for handler in tup_handlers:
        ls_handler_spies.append(mocker.spy(handler, "execute"))
    plan.execute_table_comparison(dataset_real=df_real, dataset_synthetic=df_synthetic)
    for spy_handler_execute in ls_handler_spies:
        spy_handler_execute.assert_called_once_with(resources=expected_callback_resources)
