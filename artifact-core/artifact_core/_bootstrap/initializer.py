from pathlib import Path
from typing import Optional

from artifact_core._bootstrap.libs.package_importer import PackageImporter


class ToolkitInitializer:
    @staticmethod
    def initialize(
        domain_toolkit_root: Path,
        native_artifact_path: Path,
        custom_artifact_path: Optional[Path] = None,
    ):
        if native_artifact_path is None:
            raise ValueError("Null native artifact path: edit the toolkit configuration file.")
        PackageImporter.import_all_from_package_path(
            path=native_artifact_path, root=domain_toolkit_root
        )
        if custom_artifact_path is not None:
            PackageImporter.import_all_from_package_path(path=custom_artifact_path)
