from abc import ABC
from dataclasses import dataclass
from typing import Dict, Protocol, Type, TypeVar, Union

from matplotlib.figure import Figure
from numpy import ndarray

artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound="ArtifactHyperparams")


class DataSpecProtocol(Protocol):
    pass


class NoDataSpec(DataSpecProtocol):
    pass


@dataclass(frozen=True)
class ArtifactHyperparams(ABC):
    @classmethod
    def build(cls: Type[artifactHyperparamsT], *args, **kwargs) -> artifactHyperparamsT:
        return cls(*args, **kwargs)


@dataclass(frozen=True)
class NoArtifactHyperparams(ArtifactHyperparams):
    pass


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
