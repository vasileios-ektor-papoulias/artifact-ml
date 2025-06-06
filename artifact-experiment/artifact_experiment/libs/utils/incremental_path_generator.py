import os
from typing import List, Optional


def remove_extension(filepath: str) -> str:
    root, _ = os.path.splitext(filepath)
    return root


class IncrementalPathGenerator:
    @classmethod
    def generate(cls, dir_path: str, fmt: Optional[str] = None) -> str:
        if fmt is not None:
            fmt = cls._ensure_extension(fmt=fmt)
        cls._ensure_directory(dir_path=dir_path)
        ls_existing_filepaths = os.listdir(dir_path)
        ls_indices = cls._gather_indices(ls_filepaths=ls_existing_filepaths, fmt=fmt)
        next_idx = cls._compute_next_index(ls_indices=ls_indices)
        path = cls.format_path(dir_path=dir_path, next_idx=next_idx, fmt=fmt)
        return path

    @classmethod
    def format_path(
        cls,
        dir_path: str,
        next_idx: int,
        fmt: Optional[str] = None,
    ) -> str:
        if fmt is not None:
            fmt = cls._ensure_extension(fmt=fmt)
        filename = cls._format_filename(index=next_idx, fmt=fmt)
        path = os.path.join(dir_path, filename)
        return path

    @staticmethod
    def _compute_next_index(ls_indices: List[int]) -> int:
        return max(ls_indices) + 1 if ls_indices else 0

    @classmethod
    def _gather_indices(cls, ls_filepaths: List[str], fmt: Optional[str] = None) -> List[int]:
        ls_indices: List[int] = []
        for filename in ls_filepaths:
            index = cls._get_index_from_filename(filename=filename, fmt=fmt)
            if index is not None:
                ls_indices.append(index)
        return ls_indices

    @staticmethod
    def _get_index_from_filename(filename: str, fmt: Optional[str] = None) -> Optional[int]:
        name, ext = os.path.splitext(filename)
        filename_is_valid = name.isdigit() and (
            (fmt is None and ext == "") or (fmt is not None and ext == fmt)
        )
        if filename_is_valid:
            index = int(name)
            return index

    @staticmethod
    def _format_filename(index: int, fmt: Optional[str]) -> str:
        return f"{index}{fmt}" if fmt else str(index)

    @staticmethod
    def _ensure_directory(dir_path: str):
        os.makedirs(name=dir_path, exist_ok=True)

    @staticmethod
    def _ensure_extension(fmt: str) -> str:
        if not fmt.startswith("."):
            fmt = "." + fmt
        return fmt
