from typing import Union

from artifact_core._base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.entities.file import File

TrackingData = Union[ArtifactResult, File, None]
