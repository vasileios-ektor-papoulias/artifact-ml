import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import Union


class PackageImporter:
    @staticmethod
    def import_all_from_package_path(path: Union[Path, str]):
        path = os.path.abspath(path)
        package_name = os.path.basename(path)
        parent_dir = os.path.dirname(path)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        for _, module_name, _ in pkgutil.walk_packages([path], prefix=package_name + "."):
            importlib.import_module(module_name)
