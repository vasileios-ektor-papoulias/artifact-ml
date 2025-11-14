from typing import Any, Mapping, Union

from matplotlib.figure import Figure
from numpy import floating
from numpy.typing import NDArray

Score = float
Array = NDArray[floating[Any]]
Plot = Figure
ScoreCollection = Mapping[str, Score]
ArrayCollection = Mapping[str, Array]
PlotCollection = Mapping[str, Plot]

ArtifactResult = Union[Score, Array, Plot, ScoreCollection, ArrayCollection, PlotCollection]
