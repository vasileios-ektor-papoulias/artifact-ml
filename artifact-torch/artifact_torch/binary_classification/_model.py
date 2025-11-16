from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_torch._base.model.io import ModelInput, ModelOutput
from artifact_torch._domains.classification.model import ClassificationParams, Classifier
from artifact_torch.binary_classification import BinaryClassificationResults

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
