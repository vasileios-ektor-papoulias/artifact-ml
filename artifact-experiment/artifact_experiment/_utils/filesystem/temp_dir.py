import os
import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Optional, Type, TypeVar, Union
from uuid import uuid4

PathType = Union[Path, str]

ManagedTempDirT = TypeVar("ManagedTempDirT", bound="ManagedTempDir")


class ManagedTempDir:
    def __init__(self, name: Optional[str] = None):
        self._name = name
        self._temp_dir = self._normalize_path(tempfile.mkdtemp(prefix=self.name))

    @property
    def name(self) -> str:
        return self._name if self._name is not None else ""

    @property
    def path(self) -> Path:
        return self._temp_dir

    def __enter__(self: ManagedTempDirT) -> ManagedTempDirT:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ):
        _ = exc_type
        _ = exc
        _ = tb
        self.cleanup()

    def copy_file(self, source_path: PathType) -> Path:
        source_path = self._normalize_path(path=source_path)
        dest_path = self._get_dest_path(source_path=source_path, temp_dir=self._temp_dir)
        self._copy_file(source_path=source_path, dest_path=dest_path)
        return dest_path

    def remove_file(self, path: PathType):
        path = self._normalize_path(path)
        try:
            path.relative_to(self._temp_dir)
        except ValueError:
            raise ValueError(f"Path {path} is not inside managed temp dir {self._temp_dir}")
        self._delete_file(path)

    def cleanup(self):
        self._delete_dir(dir=self._temp_dir)

    @classmethod
    def _get_dest_path(cls, source_path: Path, temp_dir: Path) -> Path:
        filename = source_path.name
        unique_filename = cls._get_unique_filename(filename=filename)
        dest_path = temp_dir / unique_filename
        return dest_path

    @staticmethod
    def _get_unique_filename(filename: str) -> str:
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid4()}{ext}"
        return unique_filename

    @staticmethod
    def _copy_file(source_path: Path, dest_path: Path):
        shutil.copy2(src=source_path, dst=dest_path)

    @staticmethod
    def _delete_file(path: Path):
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _delete_dir(dir: Path):
        if os.path.exists(dir):
            shutil.rmtree(dir)

    @staticmethod
    def _normalize_path(path: PathType) -> Path:
        return Path(path)
