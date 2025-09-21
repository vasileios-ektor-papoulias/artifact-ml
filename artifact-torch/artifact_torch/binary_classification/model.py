from abc import abstractmethod
from typing import Generic, TypeVar

import pandas as pd
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)

from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)
from artifact_torch.core.model.classifier import (
    ClassificationParams,
    Classifier,
)

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)
ClassificationParamsT = TypeVar("ClassificationParamsT", bound=ClassificationParams)


class BinaryClassifier(
    Classifier[
        ModelInputTContr,
        ModelOutputTCov,
        ClassificationParamsT,
        pd.DataFrame,
        BinaryClassificationResults,
    ],
    Generic[ModelInputTContr, ModelOutputTCov, ClassificationParamsT],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def classify(
        self, data: pd.DataFrame, params: ClassificationParamsT
    ) -> BinaryClassificationResults: ...
