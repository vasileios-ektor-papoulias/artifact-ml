from pathlib import Path
from typing import Optional

from artifact_core._bootstrap.config.toolkit_config import ToolkitConfig
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
from artifact_core._utils.system.package_importer import PackageImporter


class ToolkitInitializer:
    @staticmethod
    def load_config(domain_toolkit: DomainToolkit) -> ToolkitConfig:
        return ToolkitConfig.load(domain_toolkit=domain_toolkit)

    @classmethod
    def init_toolkit(cls, domain_toolkit: DomainToolkit, config: ToolkitConfig):
        cls._init_toolkit(
            domain_toolkit_root=domain_toolkit.root_dir,
            native_artifact_path=config.native_artifact_path,
            custom_artifact_path=config.custom_artifact_path,
        )

    @staticmethod
    def _init_toolkit(
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
