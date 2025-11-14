from typing import Hashable, Mapping, Sequence, Tuple, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")


class MapAligner:
    _error_message_thres = 5

    @classmethod
    def align(
        cls,
        left: Mapping[Hashable, T1],
        right: Mapping[Hashable, T2],
    ) -> Tuple[Sequence[Hashable], Sequence[T1], Sequence[T2]]:
        missing = [k for k in left if k not in right]
        if missing:
            raise KeyError(
                f"Right mapping missing {len(missing)} id(s): "
                f"{missing[: cls._error_message_thres]}"
                f"{'...' if len(missing) > cls._error_message_thres else ''}"
            )
        keys = list(left.keys())
        vals_left = [left[k] for k in keys]
        vals_right = [right[k] for k in keys]
        return keys, vals_left, vals_right
