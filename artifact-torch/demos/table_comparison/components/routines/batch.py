from typing import Any, List, Optional

from artifact_experiment.tracking import TrackingClient
from artifact_torch.base.components.callbacks.batch import BatchCallback
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.libs.components.callbacks.batch.loss import BatchLossCallback

from demos.table_comparison.components.protocols import DemoModelInput, DemoModelOutput
from demos.table_comparison.config.constants import BATCH_ROUTINE_PERIOD


class DemoBatchRoutine(BatchRoutine[DemoModelInput, DemoModelOutput]):
    @staticmethod
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[DemoModelInput, DemoModelOutput, Any]]:
        _ = tracking_client
        return [BatchLossCallback(period=BATCH_ROUTINE_PERIOD, tracking_client=None)]
