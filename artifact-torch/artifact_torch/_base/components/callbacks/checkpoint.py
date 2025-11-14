import os
from abc import abstractmethod
from typing import Any, Dict

import torch

from artifact_torch.base.components.callbacks.export import ExportCallback, ExportCallbackResources

CheckpointCallbackResources = ExportCallbackResources[Dict[str, Any]]


class CheckpointCallback(ExportCallback[Dict[str, Any]]):
    @classmethod
    @abstractmethod
    def _get_checkpoint_name(cls) -> str: ...

    @classmethod
    def _get_export_name(cls) -> str:
        return cls._get_checkpoint_name()

    @classmethod
    def _get_extension(cls) -> str:
        return "pth"

    @classmethod
    def _save_local(cls, data: Dict[str, Any], dir_target: str, filename: str) -> str:
        filepath = os.path.join(dir_target, filename)
        torch.save(data, filepath)
        return filepath
