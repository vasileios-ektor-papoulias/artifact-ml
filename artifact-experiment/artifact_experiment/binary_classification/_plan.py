from abc import abstractmethod
from typing import List, Mapping, Optional, Type

from artifact_core.binary_classification import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
)
from artifact_core.binary_classification.spi import (
    BinaryClassificationArtifactResources,
    BinaryClassSpecProtocol,
)
from artifact_core.typing import IdentifierType

from artifact_experiment._base.components.factories.artifact import ArtifactCallbackFactory
from artifact_experiment._base.components.plans.artifact import ArtifactPlan
from artifact_experiment._base.primitives.data_split import DataSplit
from artifact_experiment.binary_classification._callback_factory import (
    BinaryClassificationCallbackFactory,
)


class BinaryClassificationPlan(
    ArtifactPlan[
        BinaryClassificationArtifactResources,
        BinaryClassSpecProtocol,
        BinaryClassificationScoreType,
        BinaryClassificationArrayType,
        BinaryClassificationPlotType,
        BinaryClassificationScoreCollectionType,
        BinaryClassificationArrayCollectionType,
        BinaryClassificationPlotCollectionType,
    ]
):
    @staticmethod
    @abstractmethod
    def _get_score_types() -> List[BinaryClassificationScoreType]: ...

    @staticmethod
    @abstractmethod
    def _get_array_types() -> List[BinaryClassificationArrayType]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_types() -> List[BinaryClassificationPlotType]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_types() -> List[BinaryClassificationScoreCollectionType]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_types() -> List[BinaryClassificationArrayCollectionType]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_types() -> List[BinaryClassificationPlotCollectionType]: ...

    def execute_classifier_evaluation(
        self,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
        data_split: Optional[DataSplit] = None,
    ):
        artifact_resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._context.resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        super().execute_artifacts(resources=artifact_resources, data_split=data_split)

    @staticmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            BinaryClassificationArtifactResources,
            BinaryClassSpecProtocol,
            BinaryClassificationScoreType,
            BinaryClassificationArrayType,
            BinaryClassificationPlotType,
            BinaryClassificationScoreCollectionType,
            BinaryClassificationArrayCollectionType,
            BinaryClassificationPlotCollectionType,
        ]
    ]:
        return BinaryClassificationCallbackFactory
