import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import Optional, Union, overload


class PackageImporter:
    @classmethod
    def import_all_from_package_path(
        cls, path: Union[Path, str], root: Optional[Union[Path, str]] = None
    ):
        path = cls._normalize_path(path=path)
        root = cls._normalize_path(path=root)
        package_name = cls._get_package_name(path=path, root=root)
        parent_dir = cls._get_parent_directory(path=path, root=root)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        for _, module_name, _ in pkgutil.walk_packages([path], prefix=package_name + "."):
            importlib.import_module(module_name)

    @staticmethod
    def _get_package_name(path: str, root: Optional[Union[Path, str]] = None) -> str:
        if root is not None:
            root = os.path.abspath(root)
            relative_path = os.path.relpath(path, root)
            root_name = os.path.basename(root)
            return f"{root_name}.{relative_path.replace(os.sep, '.')}"
        else:
            return os.path.basename(path)

    @staticmethod
    def _get_parent_directory(path: str, root: Optional[Union[Path, str]] = None) -> str:
        if root is not None:
            return os.path.dirname(os.path.abspath(root))
        else:
            return os.path.dirname(path)

    @overload
    @staticmethod
    def _normalize_path(path: Union[Path, str]) -> str: ...

    @overload
    @staticmethod
    def _normalize_path(path: None) -> None: ...

    @staticmethod
    def _normalize_path(path: Optional[Union[Path, str]]) -> Optional[str]:
        if path is not None:
            return os.path.abspath(path)
        return None
