from abc import ABC
from dataclasses import dataclass
from typing import Dict, Protocol, Union

from matplotlib.figure import Figure
from numpy import ndarray


class DataSpecProtocol(Protocol):
    pass


class NoDataSpec(DataSpecProtocol):
    pass


@dataclass(frozen=True)
class ArtifactHyperparams(ABC):
    pass


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
