import torch
from artifact_experiment.spi.callbacks import ExportCallback
from artifact_experiment.tracking.spi import FileWriter

from artifact_torch._base.components.resources.checkpoint import CheckpointCallbackResources
from artifact_torch._utils.filesystem.filename_appender import FilenameAppender
from artifact_torch._utils.scheduling.periodic_actions import PeriodicActionTrigger


class CheckpointCallback(ExportCallback[CheckpointCallbackResources]):
    _export_name = "TRAINING_CHECKPOINTS"

    def __init__(self, period: int, writer: FileWriter):
        super().__init__(writer=writer)
        self._period = period

    @classmethod
    def _get_export_name(cls) -> str:
        return cls._export_name

    @classmethod
    def _get_extension(cls) -> str:
        return "pth"

    def execute(self, resources: CheckpointCallbackResources):
        if PeriodicActionTrigger.should_trigger(step=resources.epoch, period=self._period):
            super().execute(resources=resources)

    @classmethod
    def _save_local(cls, resources: CheckpointCallbackResources, filepath: str) -> str:
        suffix = cls._get_filename_suffix(epoch=resources.epoch)
        filepath = FilenameAppender.append(filepath=filepath, text=suffix).as_posix()
        torch.save(resources.export_data, filepath)
        return filepath

    @staticmethod
    def _get_filename_suffix(epoch: int) -> str:
        return f"_EPOCH_{epoch}"
