from typing import Any, Iterable

import torch


class TensorFlattener:
    @classmethod
    def flatten_tensors(cls, x: Any) -> Iterable[torch.Tensor]:
        if torch.is_tensor(x):
            yield x
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from cls.flatten_tensors(y)
        elif isinstance(x, dict):
            for y in x.values():
                yield from cls.flatten_tensors(y)
