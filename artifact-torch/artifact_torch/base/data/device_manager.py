from typing import Any, Dict, List, Tuple, overload

import torch

from artifact_torch.base.model.io import ModelIO


class DeviceManager:
    @classmethod
    @overload
    def move_to_device(cls, data: ModelIO, device: torch.device) -> ModelIO: ...

    @classmethod
    @overload
    def move_to_device(cls, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]: ...

    @classmethod
    @overload
    def move_to_device(cls, data: List[Any], device: torch.device) -> List[Any]: ...

    @classmethod
    @overload
    def move_to_device(cls, data: Tuple[Any, ...], device: torch.device) -> Tuple[Any, ...]: ...

    @classmethod
    @overload
    def move_to_device(cls, data: torch.Tensor, device: torch.device) -> torch.Tensor: ...

    @classmethod
    @overload
    def move_to_device(cls, data: Any, device: torch.device) -> Any: ...

    @classmethod
    def move_to_device(cls, data: Any, device: torch.device) -> Any:
        if isinstance(data, dict):
            return {k: cls.move_to_device(data=v, device=device) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.move_to_device(data=item, device=device) for item in data]
        elif isinstance(data, tuple):
            return tuple(cls.move_to_device(data=item, device=device) for item in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data
