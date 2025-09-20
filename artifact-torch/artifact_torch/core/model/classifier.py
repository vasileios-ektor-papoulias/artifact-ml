from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.classification.classification_results import ClassificationResults

from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

ModelInputTContr = TypeVar("ModelInputTContr", bound="ModelInput", contravariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound="ModelOutput", covariant=True)
ClassificationParamsT = TypeVar("ClassificationParamsT", bound="ClassificationParams")
ClassificationDataT = TypeVar("ClassificationDataT")
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)


@dataclass
class ClassificationParams:
    pass


class Classifier(
    Model[ModelInputTContr, ModelOutputTCov],
    Generic[
        ModelInputTContr,
        ModelOutputTCov,
        CategoricalFeatureSpecProtocolT,
        ClassificationParamsT,
        ClassificationDataT,
    ],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def classify(
        self, data: ClassificationDataT, params: ClassificationParamsT
    ) -> ClassificationResults[CategoricalFeatureSpecProtocolT]: ...
