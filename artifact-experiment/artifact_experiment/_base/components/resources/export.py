from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

from artifact_experiment._base.components.resources.tracking import TrackingCallbackResources

ExportDataTCov = TypeVar("ExportDataTCov", covariant=True)


@dataclass(frozen=True)
class ExportCallbackResources(TrackingCallbackResources, Generic[ExportDataTCov]):
    _: KW_ONLY
    export_data: ExportDataTCov
