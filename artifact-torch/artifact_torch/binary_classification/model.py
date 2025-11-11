from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core._libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)

from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.core.model.classifier import ClassificationParams, Classifier

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)
ClassificationParamsTContr = TypeVar(
    "ClassificationParamsTContr", bound=ClassificationParams, contravariant=True
)
ClassificationDataTContr = TypeVar("ClassificationDataTContr", contravariant=True)


class BinaryClassifier(
    Classifier[
        ModelInputTContr,
        ModelOutputTCov,
        ClassificationParamsTContr,
        ClassificationDataTContr,
        BinaryClassificationResults,
    ],
    Generic[
        ModelInputTContr, ModelOutputTCov, ClassificationParamsTContr, ClassificationDataTContr
    ],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def classify(
        self, data: ClassificationDataTContr, params: ClassificationParamsTContr
    ) -> BinaryClassificationResults: ...
