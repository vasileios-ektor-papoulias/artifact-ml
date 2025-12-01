from pathlib import Path
from typing import Optional

from artifact_core._bootstrap.config.override_locator import ConfigOverrideLocator
from artifact_core._bootstrap.config.toolkit_config import ToolkitConfig
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
from artifact_core._utils.system.module_importer import ModuleImporter


class ToolkitInitializer:
    _config_override_dir: Optional[Path] = None

    @classmethod
    def init_toolkit(cls, domain_toolkit: DomainToolkit):
        config = cls.load_toolkit_config(domain_toolkit=domain_toolkit)
        ModuleImporter.import_modules(
            path=domain_toolkit.native_artifacts_dir, root=domain_toolkit.package_root
        )
        if cls._config_override_dir is not None and config.custom_artifacts_dir is not None:
            ModuleImporter.import_modules(
                path=config.custom_artifacts_dir, root=cls._config_override_dir.parent
            )

    @classmethod
    def load_toolkit_config(cls, domain_toolkit: DomainToolkit) -> ToolkitConfig:
        if cls._config_override_dir is None:
            cls._config_override_dir = cls._locate_config_override()
        return ToolkitConfig.load(
            domain_toolkit=domain_toolkit, config_override_dir=cls._config_override_dir
        )

    @staticmethod
    def _resolve_custom_artifacts_dir(
        user_override_dir: Optional[Path], relative_path: Optional[Path]
    ) -> Optional[Path]:
        if relative_path and user_override_dir:
            return (user_override_dir.parent / relative_path).resolve()

    @staticmethod
    def _locate_config_override() -> Optional[Path]:
        return ConfigOverrideLocator.find()
