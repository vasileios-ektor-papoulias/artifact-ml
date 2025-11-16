from typing import Any, TypeVar

from artifact_torch._base.components.callbacks.hook import HookCallbackResources
from artifact_torch._base.model.base import Model
from artifact_torch._base.model.io import ModelOutput

ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)


ModelIOCallbackResources = HookCallbackResources[Model[Any, ModelOutputTCov]]
