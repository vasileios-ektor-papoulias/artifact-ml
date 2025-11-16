import os
from typing import List, Optional

from artifact_experiment._utils.filesystem.extension_normalizer import ExtensionNormalizer


class IncrementalPathFormatter:
    @classmethod
    def format(
        cls,
        dir_path: str,
        next_idx: int,
        extension: Optional[str] = None,
    ) -> str:
        if extension is not None:
            extension = ExtensionNormalizer.normalize(extension=extension)
        filename = cls._format_filename(index=next_idx, fmt=extension)
        path = os.path.join(dir_path, filename)
        return path

    @staticmethod
    def _format_filename(index: int, fmt: Optional[str]) -> str:
        return f"{index}{fmt}" if fmt else str(index)


class IncrementalPathGenerator:
    @classmethod
    def generate(cls, dir_path: str, ext: Optional[str] = None) -> str:
        if ext is not None:
            ext = ExtensionNormalizer.normalize(extension=ext)
        cls._ensure_directory(dir_path=dir_path)
        ls_filenames = os.listdir(dir_path)
        ls_indices = cls._gather_indices(ls_filenames=ls_filenames, fmt=ext)
        next_idx = cls._compute_next_index(ls_indices=ls_indices)
        path = IncrementalPathFormatter.format(dir_path=dir_path, next_idx=next_idx, extension=ext)
        return path

    @staticmethod
    def _compute_next_index(ls_indices: List[int]) -> int:
        return max(ls_indices) + 1 if ls_indices else 0

    @classmethod
    def _gather_indices(cls, ls_filenames: List[str], fmt: Optional[str] = None) -> List[int]:
        ls_indices: List[int] = []
        for filename in ls_filenames:
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
    def _ensure_directory(dir_path: str):
        os.makedirs(name=dir_path, exist_ok=True)
