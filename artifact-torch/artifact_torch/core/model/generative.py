from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

ModelInputTContr = TypeVar("ModelInputTContr", bound="ModelInput", contravariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound="ModelOutput", covariant=True)
GenerationParamsT = TypeVar("GenerationParamsT", bound="GenerationParams")
GenerationT = TypeVar("GenerationT")


@dataclass
class GenerationParams:
    pass


class GenerativeModel(
    Model[ModelInputTContr, ModelOutputTCov],
    Generic[ModelInputTContr, ModelOutputTCov, GenerationParamsT, GenerationT],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def generate(self, params: GenerationParamsT) -> GenerationT: ...
