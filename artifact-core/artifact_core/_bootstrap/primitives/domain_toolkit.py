from enum import Enum
from pathlib import Path


class DomainToolkit(Enum):
    TABLE_COMPARISON = "TABLE_COMPARISON"
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"

    @property
    def toolkit_name(self) -> str:
        if self is DomainToolkit.TABLE_COMPARISON:
            return "table_comparison"
        elif self is DomainToolkit.BINARY_CLASSIFICATION:
            return "binary_classification"
        else:
            raise ValueError(f"Unrecognized domain toolkit: {self}")

    @property
    def package_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def toolkit_root(self) -> Path:
        return self.package_root / self.toolkit_name

    @property
    def native_artifacts_dir(self) -> Path:
        return self.toolkit_root / "_artifacts"

    @property
    def base_config_filepath(self) -> Path:
        return self.toolkit_root / "_config" / "raw.json"

    @property
    def config_override_filename(self) -> str:
        return f"{self.toolkit_name}.json"
