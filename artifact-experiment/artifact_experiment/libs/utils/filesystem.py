import os
from typing import List, Optional


class IncrementalPathGenerator:
    @classmethod
    def generate(cls, dir_path: str, fmt: Optional[str] = None) -> str:
        cls._ensure_directory(dir_path=dir_path)
        ls_existing_filepaths = cls._get_existing_files_fs(dir_path=dir_path)
        path = cls.generate_from_existing_filepaths(
            ls_existing_filepaths=ls_existing_filepaths, dir_local=dir_path, fmt=fmt
        )
        return path

    @classmethod
    def generate_from_existing_filepaths(
        cls,
        ls_existing_filepaths: List[str],
        dir_local: str,
        fmt: Optional[str] = None,
    ) -> str:
        ls_indices = cls._gather_indices(ls_existing_filepaths=ls_existing_filepaths, fmt=fmt)
        next_idx = cls._compute_next_index(ls_indices=ls_indices)
        filename = cls._format_filename(index=next_idx, fmt=fmt)
        path = os.path.join(dir_local, filename)
        return path

    @staticmethod
    def _gather_indices(ls_existing_filepaths: List[str], fmt: Optional[str] = None) -> List[int]:
        ls_indices: List[int] = []
        for fname in ls_existing_filepaths:
            name, ext = os.path.splitext(fname)
            ext = ext.lstrip(".")
            if name.isdigit() and ((fmt is None and ext == "") or (fmt is not None and ext == fmt)):
                ls_indices.append(int(name))
        return ls_indices

    @staticmethod
    def _get_existing_files_fs(dir_path: str) -> List[str]:
        ls_existing_filepaths = os.listdir(dir_path)
        return ls_existing_filepaths

    @staticmethod
    def _compute_next_index(ls_indices: List[int]) -> int:
        return max(ls_indices) + 1 if ls_indices else 0

    @staticmethod
    def _format_filename(index: int, fmt: Optional[str]) -> str:
        return f"{index}.{fmt}" if fmt else str(index)

    @staticmethod
    def _ensure_directory(dir_path: str):
        os.makedirs(name=dir_path, exist_ok=True)
