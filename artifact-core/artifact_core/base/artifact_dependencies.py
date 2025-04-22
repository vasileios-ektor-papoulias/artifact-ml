from abc import ABC
from dataclasses import dataclass
from typing import Dict, Protocol, Type, TypeVar, Union

from matplotlib.figure import Figure
from numpy import ndarray

artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound="ArtifactHyperparams")


@dataclass(frozen=True)
class ArtifactHyperparams(ABC):
    @classmethod
    def build(cls: Type[artifactHyperparamsT], *args, **kwargs) -> artifactHyperparamsT:
        return cls(*args, **kwargs)


@dataclass(frozen=True)
class NoArtifactHyperparams(ArtifactHyperparams):
    pass


NO_ARTIFACT_HYPERPARAMS = NoArtifactHyperparams()


class ResourceSpecProtocol(Protocol):
    pass


class NoResourceSpec(ResourceSpecProtocol):
    pass


NO_RESOURCE_SPEC = NoResourceSpec()


@dataclass(frozen=True)
class ArtifactResources:
    pass


ArtifactResult = Union[
    float,
    ndarray,
    Figure,
    Dict[str, float],
    Dict[str, ndarray],
    Dict[str, Figure],
]
