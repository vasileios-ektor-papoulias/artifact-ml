import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import Optional, Union


class PackageImporter:
    @classmethod
    def import_all_from_package_path(
        cls, path: Union[Path, str], root: Optional[Union[Path, str]] = None
    ):
        path = os.path.abspath(path)
        package_name = cls._get_package_name(path, root)
        parent_dir = cls._get_parent_directory(path, root)
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
