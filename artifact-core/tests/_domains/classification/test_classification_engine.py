from typing import Any, Callable, Dict, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_core._domains.classification.resources import ClassificationArtifactResources
from artifact_core._utils.collections.entity_store import IdentifierType
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests._domains.classification.conftest import MakeClassificationResults, MakeClassStore
from tests._domains.classification.dummy.artifacts.base import DummyClassificationArtifact
from tests._domains.classification.dummy.engine.engine import DummyClassificationEngine
from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.resource_spec import DummyClassSpec
from tests._domains.classification.dummy.types.array_collections import (
    DummyClassificationArrayCollectionType,
)
from tests._domains.classification.dummy.types.arrays import DummyClassificationArrayType
from tests._domains.classification.dummy.types.plot_collections import (
    DummyClassificationPlotCollectionType,
)
from tests._domains.classification.dummy.types.plots import DummyClassificationPlotType
from tests._domains.classification.dummy.types.score_collections import (
    DummyClassificationScoreCollectionType,
)
from tests._domains.classification.dummy.types.scores import DummyClassificationScoreType


@pytest.fixture
def id_to_class_idx() -> Dict[IdentifierType, int]:
    return {0: 0, 1: 1}


@pytest.fixture
def id_to_predicted_class_idx() -> Dict[IdentifierType, int]:
    return {0: 0, 1: 1}


@pytest.fixture
def engine_factory(
    mocker: MockerFixture,
) -> Callable[
    [str, DummyClassSpec, ArtifactResult],
    Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine],
]:
    def _factory(
        modality: str,
        resource_spec: DummyClassSpec,
        return_value: ArtifactResult,
    ) -> Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine]:
        artifact = mocker.Mock(name="artifact", spec_set=DummyClassificationArtifact[Any, Any])
        registry = mocker.Mock(name="registry", spec_set=DummyClassificationRegistry[Any, Any])
        other_registry = mocker.Mock(
            name="other_registry", spec_set=DummyClassificationRegistry[Any, Any]
        )
        artifact.compute.return_value = return_value
        registry.get.return_value = artifact
        registries = {
            f"{modality}_registry": other_registry
            for modality in [
                "score",
                "array",
                "plot",
                "score_collection",
                "array_collection",
                "plot_collection",
            ]
        }
        registries[f"{modality}_registry"] = registry
        engine = DummyClassificationEngine(resource_spec=resource_spec, **registries)
        return artifact, registry, other_registry, engine

    return _factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, expected_result",
    [
        (
            "SCORE_TYPE_1",
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            1,
        ),
        (
            "SCORE_TYPE_2",
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            2,
        ),
        (
            "CUSTOM_SCORE",
            DummyClassSpec(class_names=["x", "y"], label_name="class"),
            3,
        ),
    ],
)
def test_produce_classification_score(
    engine_factory: Callable[
        [str, DummyClassSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine],
    ],
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    artifact_type: Union[DummyClassificationScoreType, str],
    resource_spec: DummyClassSpec,
    expected_result: Score,
):
    artifact, registry, other_registry, engine = engine_factory(
        "score", resource_spec, expected_result
    )
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    resources = ClassificationArtifactResources(
        true_class_store=true_class_store, classification_results=classification_results
    )
    result = engine.produce_classification_score(
        score_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, expected_result",
    [
        (
            "ARRAY_TYPE_1",
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            np.array([1]),
        ),
        (
            "ARRAY_TYPE_2",
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            np.array([2]),
        ),
        (
            "CUSTOM_ARRAY",
            DummyClassSpec(class_names=["x", "y"], label_name="class"),
            np.array([3]),
        ),
    ],
)
def test_produce_classification_array(
    engine_factory: Callable[
        [str, DummyClassSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine],
    ],
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    artifact_type: Union[DummyClassificationArrayType, str],
    resource_spec: DummyClassSpec,
    expected_result: Array,
):
    artifact, registry, other_registry, engine = engine_factory(
        "array", resource_spec, expected_result
    )
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    resources = ClassificationArtifactResources(
        true_class_store=true_class_store, classification_results=classification_results
    )
    result = engine.produce_classification_array(
        array_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert (result == expected_result).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, expected_result",
    [
        (
            "PLOT_TYPE_1",
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            Figure(),
        ),
        (
            "PLOT_TYPE_2",
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            Figure(),
        ),
        (
            "CUSTOM_PLOT",
            DummyClassSpec(class_names=["x", "y"], label_name="class"),
            Figure(),
        ),
    ],
)
def test_produce_classification_plot(
    engine_factory: Callable[
        [str, DummyClassSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine],
    ],
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    artifact_type: Union[DummyClassificationPlotType, str],
    resource_spec: DummyClassSpec,
    expected_result: Plot,
):
    artifact, registry, other_registry, engine = engine_factory(
        "plot", resource_spec, expected_result
    )
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    resources = ClassificationArtifactResources(
        true_class_store=true_class_store, classification_results=classification_results
    )
    result = engine.produce_classification_plot(
        plot_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, expected_result",
    [
        (
            "SCORE_COLLECTION_TYPE_1",
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {"item": 1},
        ),
        (
            "SCORE_COLLECTION_TYPE_2",
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            {"item": 2},
        ),
        (
            "CUSTOM_SCORE_COLLECTION",
            DummyClassSpec(class_names=["x", "y"], label_name="class"),
            {"item": 3},
        ),
    ],
)
def test_produce_classification_score_collection(
    engine_factory: Callable[
        [str, DummyClassSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine],
    ],
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    artifact_type: Union[DummyClassificationScoreCollectionType, str],
    resource_spec: DummyClassSpec,
    expected_result: ScoreCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "score_collection", resource_spec, expected_result
    )
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    resources = ClassificationArtifactResources(
        true_class_store=true_class_store, classification_results=classification_results
    )
    result = engine.produce_classification_score_collection(
        score_collection_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, expected_result",
    [
        (
            "ARRAY_COLLECTION_TYPE_1",
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {"item": np.array([1])},
        ),
        (
            "ARRAY_COLLECTION_TYPE_2",
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            {"item": np.array([2])},
        ),
        (
            "CUSTOM_ARRAY_COLLECTION",
            DummyClassSpec(class_names=["x", "y"], label_name="class"),
            {"item": np.array([3])},
        ),
    ],
)
def test_produce_classification_array_collection(
    engine_factory: Callable[
        [str, DummyClassSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine],
    ],
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    artifact_type: Union[DummyClassificationArrayCollectionType, str],
    resource_spec: DummyClassSpec,
    expected_result: ArrayCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "array_collection", resource_spec, expected_result
    )
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    resources = ClassificationArtifactResources(
        true_class_store=true_class_store, classification_results=classification_results
    )
    result = engine.produce_classification_array_collection(
        array_collection_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    for array_name in result.keys():
        assert (result[array_name] == expected_result[array_name]).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, expected_result",
    [
        (
            "PLOT_COLLECTION_TYPE_1",
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {"item": Figure()},
        ),
        (
            "PLOT_COLLECTION_TYPE_2",
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            {"item": Figure()},
        ),
        (
            "CUSTOM_PLOT_COLLECTION",
            DummyClassSpec(class_names=["x", "y"], label_name="class"),
            {"item": Figure()},
        ),
    ],
)
def test_produce_classification_plot_collection(
    engine_factory: Callable[
        [str, DummyClassSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyClassificationEngine],
    ],
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    artifact_type: Union[DummyClassificationPlotCollectionType, str],
    resource_spec: DummyClassSpec,
    expected_result: PlotCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "plot_collection", resource_spec, expected_result
    )
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    resources = ClassificationArtifactResources(
        true_class_store=true_class_store, classification_results=classification_results
    )
    result = engine.produce_classification_plot_collection(
        plot_collection_type=artifact_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result
