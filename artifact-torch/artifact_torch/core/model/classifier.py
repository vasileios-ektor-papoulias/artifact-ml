from abc import abstractmethod
from typing import Generic, TypedDict, TypeVar

from artifact_core._libs.resourcess.classification.classification_results import (
    ClassificationResults,
)

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
ClassificationDataTContr = TypeVar("ClassificationDataTContr", contravariant=True)
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
        ClassificationDataTContr,
        ClassificationResultsTCov,
    ],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def classify(
        self, data: ClassificationDataTContr, params: ClassificationParamsTContr
    ) -> ClassificationResultsTCov: ...
