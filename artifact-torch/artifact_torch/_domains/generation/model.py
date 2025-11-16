from abc import abstractmethod
from typing import Generic, TypedDict, TypeVar

from artifact_torch._base.model.base import Model
from artifact_torch._base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound="ModelInput", contravariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound="ModelOutput", covariant=True)
GenerationParamsTContr = TypeVar(
    "GenerationParamsTContr", bound="GenerationParams", contravariant=True
)
GenerationTCov = TypeVar("GenerationTCov", covariant=True)


class GenerationParams(TypedDict):
    pass


class GenerativeModel(
    Model[ModelInputTContr, ModelOutputTCov],
    Generic[ModelInputTContr, ModelOutputTCov, GenerationParamsTContr, GenerationTCov],
):
    @abstractmethod
    def forward(self, model_input: ModelInputTContr, *args, **kwargs) -> ModelOutputTCov: ...

    @abstractmethod
    def generate(self, params: GenerationParamsTContr) -> GenerationTCov: ...
