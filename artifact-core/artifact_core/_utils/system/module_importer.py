import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import Union

from artifact_core._utils.filesystem.path_normalizer import PathResolver
from artifact_core._utils.filesystem.separator_replacer import SeparatorReplacer


class ModuleImporter:
    @classmethod
    def import_modules(cls, path: Union[Path, str], root: Union[Path, str]):
        root = PathResolver.resolve(path=root)
        path = PathResolver.resolve(path=path)
        parent_dir = cls._get_parent_dir(root=root)
        module_path = cls._get_module_path(path=path, root=root)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        for _, module_name, _ in pkgutil.walk_packages([path], prefix=module_path + "."):
            importlib.import_module(name=module_name)

    @staticmethod
    def _get_module_path(path: Union[Path, str], root: Union[Path, str]) -> str:
        relative_path = os.path.relpath(path, root)
        root_package = os.path.basename(root)
        module_path = os.path.join(root_package, relative_path)
        module_path = SeparatorReplacer.replace_separator(path=module_path, new=".")
        return module_path

    @staticmethod
    def _get_parent_dir(root: Union[Path, str]) -> str:
        return os.path.dirname(os.path.abspath(root))
