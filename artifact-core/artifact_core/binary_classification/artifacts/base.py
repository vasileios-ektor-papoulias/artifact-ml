from abc import abstractmethod
from typing import Dict, Generic, Tuple, TypeVar

import pandas as pd
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResult,
)
from artifact_core.core.classification.artifact import (
    ClassificationArtifact,
    ClassificationArtifactResources,
)
from artifact_core.libs.resource_spec.labels.binary import BinaryLabelsSpec
from artifact_core.libs.resource_validation.labels.label_validator import LabelValidator

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound="ArtifactHyperparams")


class BinaryClassificationArtifact(
    ClassificationArtifact[
        pd.DataFrame,
        ArtifactResultT,
        ArtifactHyperparamsT,
        BinaryLabelsSpec,
    ],
    Generic[ArtifactResultT, ArtifactHyperparamsT],
):
    @abstractmethod
    def _evaluate_classification(
        self, labels_ground_truth: pd.DataFrame, labels_predicted: pd.DataFrame
    ) -> ArtifactResultT: ...

    def _validate_labels(
        self, labels_ground_truth: pd.DataFrame, labels_predicted: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labels_ground_truth_validated = LabelValidator.validate(
            df_labels=labels_ground_truth,
            id_col=self._resource_spec.id_name,
            label_classes_map=self._resource_spec.label_classes_map,
        )
        labels_predicted_validated = LabelValidator.validate(
            df_labels=labels_predicted,
            id_col=self._resource_spec.id_name,
            label_classes_map=self._resource_spec.label_classes_map,
        )
        return labels_ground_truth_validated, labels_predicted_validated


BinaryClassificationArtifactResources = ClassificationArtifactResources[pd.DataFrame]

BinaryClassificationScore = BinaryClassificationArtifact[float, ArtifactHyperparamsT]
BinaryClassificationArray = BinaryClassificationArtifact[ndarray, ArtifactHyperparamsT]
BinaryClassificationPlot = BinaryClassificationArtifact[Figure, ArtifactHyperparamsT]
BinaryClassificationScoreCollection = BinaryClassificationArtifact[
    Dict[str, float], ArtifactHyperparamsT
]
BinaryClassificationArrayCollection = BinaryClassificationArtifact[
    Dict[str, ndarray], ArtifactHyperparamsT
]
BinaryClassificationPlotCollection = BinaryClassificationArtifact[
    Dict[str, Figure], ArtifactHyperparamsT
]
