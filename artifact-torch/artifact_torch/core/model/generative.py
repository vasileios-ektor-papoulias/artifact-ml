from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
ModelOutputT = TypeVar("ModelOutputT", bound="ModelOutput")
GenerationParamsT = TypeVar("GenerationParamsT", bound="GenerationParams")
GenerationT = TypeVar("GenerationT")


@dataclass
class GenerationParams:
    pass


class GenerativeModel(
    Model[ModelInputT, ModelOutputT],
    Generic[ModelInputT, ModelOutputT, GenerationParamsT, GenerationT],
):
    @abstractmethod
    def forward(self, model_input: ModelInputT, *args, **kwargs) -> ModelOutputT: ...

    @abstractmethod
    def generate(self, params: GenerationParamsT) -> GenerationT: ...
