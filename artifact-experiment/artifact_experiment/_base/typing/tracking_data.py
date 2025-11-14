from typing import Union

from artifact_core.typing import ArtifactResult

from artifact_experiment._base.primitives.file import File

TrackingData = Union[ArtifactResult, File, None]
