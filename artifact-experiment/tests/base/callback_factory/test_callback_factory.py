import pytest
from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from pytest_mock import MockerFixture

from tests.base.callback_factory.dummy.callback_factory import DummyCallbackFactory
from tests.base.dummy_artifact_toolkit import (
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyArrayType,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
    DummyPlotType,
    DummyResourceSpec,
    DummyScoreCollectionRegistry,
    DummyScoreCollectionType,
    DummyScoreRegistry,
    DummyScoreType,
)


@pytest.fixture
def dummy_resource_spec() -> DummyResourceSpec:
    return DummyResourceSpec()


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "score_type",
    [DummyScoreType.DUMMY_SCORE_1],
)
def test_build_score_callback(
    mocker: MockerFixture, dummy_resource_spec: DummyResourceSpec, score_type: DummyScoreType
):
    spy_get = mocker.spy(DummyScoreRegistry, "get")
    callback = DummyCallbackFactory.build_score_callback(
        score_type=score_type, resource_spec=dummy_resource_spec
    )
    spy_get.assert_called_once_with(artifact_type=score_type, resource_spec=dummy_resource_spec)
    assert isinstance(callback, ArtifactScoreCallback)
    assert callback.key == score_type.name


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "array_type",
    [DummyArrayType.DUMMY_ARRAY_1],
)
def test_build_array_callback(
    mocker: MockerFixture, dummy_resource_spec: DummyResourceSpec, array_type: DummyArrayType
):
    spy_get = mocker.spy(DummyArrayRegistry, "get")
    callback = DummyCallbackFactory.build_array_callback(
        array_type=array_type, resource_spec=dummy_resource_spec
    )
    spy_get.assert_called_once_with(artifact_type=array_type, resource_spec=dummy_resource_spec)
    assert isinstance(callback, ArtifactArrayCallback)
    assert callback.key == array_type.name


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "plot_type",
    [DummyPlotType.DUMMY_PLOT_1],
)
def test_build_plot_callback(
    mocker: MockerFixture, dummy_resource_spec: DummyResourceSpec, plot_type: DummyPlotType
):
    spy_get = mocker.spy(DummyPlotRegistry, "get")
    callback = DummyCallbackFactory.build_plot_callback(
        plot_type=plot_type, resource_spec=dummy_resource_spec
    )
    spy_get.assert_called_once_with(artifact_type=plot_type, resource_spec=dummy_resource_spec)
    assert isinstance(callback, ArtifactPlotCallback)
    assert callback.key == plot_type.name


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "score_collection_type",
    [DummyScoreCollectionType.DUMMY_SCORE_COLLECTION_1],
)
def test_build_score_collection_callback(
    mocker: MockerFixture,
    dummy_resource_spec: DummyResourceSpec,
    score_collection_type: DummyScoreCollectionType,
):
    spy_get = mocker.spy(DummyScoreCollectionRegistry, "get")
    callback = DummyCallbackFactory.build_score_collection_callback(
        score_collection_type=score_collection_type, resource_spec=dummy_resource_spec
    )
    spy_get.assert_called_once_with(
        artifact_type=score_collection_type, resource_spec=dummy_resource_spec
    )
    assert isinstance(callback, ArtifactScoreCollectionCallback)
    assert callback.key == score_collection_type.name


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "array_collection_type",
    [DummyArrayCollectionType.DUMMY_ARRAY_COLLECTION_1],
)
def test_build_array_collection_callback(
    mocker: MockerFixture,
    dummy_resource_spec: DummyResourceSpec,
    array_collection_type: DummyArrayCollectionType,
):
    spy_get = mocker.spy(DummyArrayCollectionRegistry, "get")
    callback = DummyCallbackFactory.build_array_collection_callback(
        array_collection_type=array_collection_type, resource_spec=dummy_resource_spec
    )
    spy_get.assert_called_once_with(
        artifact_type=array_collection_type, resource_spec=dummy_resource_spec
    )
    assert isinstance(callback, ArtifactArrayCollectionCallback)
    assert callback.key == array_collection_type.name


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "plot_collection_type",
    [DummyPlotCollectionType.DUMMY_PLOT_COLLECTION_1],
)
def test_build_plot_collection_callback(
    mocker: MockerFixture,
    dummy_resource_spec: DummyResourceSpec,
    plot_collection_type: DummyPlotCollectionType,
):
    spy_get = mocker.spy(DummyPlotCollectionRegistry, "get")
    callback = DummyCallbackFactory.build_plot_collection_callback(
        plot_collection_type=plot_collection_type, resource_spec=dummy_resource_spec
    )
    spy_get.assert_called_once_with(
        artifact_type=plot_collection_type, resource_spec=dummy_resource_spec
    )
    assert isinstance(callback, ArtifactPlotCollectionCallback)
    assert callback.key == plot_collection_type.name
