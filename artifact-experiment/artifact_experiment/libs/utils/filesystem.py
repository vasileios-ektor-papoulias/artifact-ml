import os
from typing import List, Optional


class IncrementalPathGenerator:
    @classmethod
    def generate(cls, dir_path: str, fmt: Optional[str] = None) -> str:
        cls._ensure_directory(dir_path)
        indices = cls._gather_indices(dir_path, fmt)
        next_idx = cls._compute_next_index(indices)
        filename = cls._format_filename(next_idx, fmt)
        return os.path.join(dir_path, filename)

    @staticmethod
    def _ensure_directory(dir_path: str):
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def _gather_indices(dir_path: str, fmt: Optional[str] = None) -> List[int]:
        existing_files = os.listdir(dir_path)
        indices: list[int] = []
        for fname in existing_files:
            name, ext = os.path.splitext(fname)
            ext = ext.lstrip(".")
            if name.isdigit() and ((fmt is None and ext == "") or (fmt is not None and ext == fmt)):
                indices.append(int(name))
        return indices

    @staticmethod
    def _compute_next_index(indices: list[int]) -> int:
        return max(indices) + 1 if indices else 0

    @staticmethod
    def _format_filename(index: int, fmt: Optional[str]) -> str:
        return f"{index}.{fmt}" if fmt else str(index)
