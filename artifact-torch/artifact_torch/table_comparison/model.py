from abc import abstractmethod
from typing import Generic, TypeVar

import pandas as pd

from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)
from artifact_torch.core.model.generative import GenerationParams, GenerativeModel

ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
ModelOutputT = TypeVar("ModelOutputT", bound="ModelOutput")
GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)


class TabularGenerativeModel(
    GenerativeModel[ModelInputT, ModelOutputT, GenerationParamsT, pd.DataFrame],
    Generic[ModelInputT, ModelOutputT, GenerationParamsT],
):
    @abstractmethod
    def forward(self, model_input: ModelInputT, *args, **kwargs) -> ModelOutputT: ...

    @abstractmethod
    def generate(self, params: GenerationParamsT) -> pd.DataFrame: ...
