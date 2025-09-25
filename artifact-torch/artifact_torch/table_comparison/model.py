from abc import abstractmethod
from typing import Generic, TypeVar

import pandas as pd

from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)
from artifact_torch.core.model.generative import GenerationParams, GenerativeModel

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)
GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)


class TableSynthesizer(
    GenerativeModel[ModelInputTContr, ModelOutputTCov, GenerationParamsT, pd.DataFrame],
    Generic[ModelInputTContr, ModelOutputTCov, GenerationParamsT],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def generate(self, params: GenerationParamsT) -> pd.DataFrame: ...
