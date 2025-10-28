from abc import abstractmethod
from typing import Generic, TypedDict, TypeVar

from artifact_core.libs.resources.classification.classification_results import ClassificationResults

from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

ModelInputTContr = TypeVar("ModelInputTContr", bound="ModelInput", contravariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound="ModelOutput", covariant=True)
ClassificationParamsTContr = TypeVar(
    "ClassificationParamsTContr", bound="ClassificationParams", contravariant=True
)
ClassificationDataT = TypeVar("ClassificationDataT")
ClassificationResultsTCov = TypeVar(
    "ClassificationResultsTCov", bound=ClassificationResults, covariant=True
)


class ClassificationParams(TypedDict):
    pass


class Classifier(
    Model[ModelInputTContr, ModelOutputTCov],
    Generic[
        ModelInputTContr,
        ModelOutputTCov,
        ClassificationParamsTContr,
        ClassificationDataT,
        ClassificationResultsTCov,
    ],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def classify(
        self, data: ClassificationDataT, params: ClassificationParamsTContr
    ) -> ClassificationResultsTCov: ...
