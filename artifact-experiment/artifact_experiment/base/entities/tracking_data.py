from typing import Union

from artifact_core.interfaces.types import ArtifactResult

from artifact_experiment.base.entities.file import File

TrackingData = Union[ArtifactResult, File, None]
