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
GenerationParamsTContr = TypeVar(
    "GenerationParamsTContr", bound=GenerationParams, contravariant=True
)


class TableSynthesizer(
    GenerativeModel[ModelInputTContr, ModelOutputTCov, GenerationParamsTContr, pd.DataFrame],
    Generic[ModelInputTContr, ModelOutputTCov, GenerationParamsTContr],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def generate(self, params: GenerationParamsTContr) -> pd.DataFrame: ...
