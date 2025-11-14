from abc import ABC
from dataclasses import dataclass
from typing import Type, TypeVar

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound="ArtifactHyperparams")


@dataclass(frozen=True)
class ArtifactHyperparams(ABC):
    @classmethod
    def build(cls: Type[ArtifactHyperparamsT], *args, **kwargs) -> ArtifactHyperparamsT:
        return cls(*args, **kwargs)


@dataclass(frozen=True)
class NoArtifactHyperparams(ArtifactHyperparams):
    pass


NO_ARTIFACT_HYPERPARAMS = NoArtifactHyperparams()
