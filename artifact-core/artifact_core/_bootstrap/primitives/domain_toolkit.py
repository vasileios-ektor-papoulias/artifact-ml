from enum import Enum
from pathlib import Path


class DomainToolkit(Enum):
    TABLE_COMPARISON = "TABLE_COMPARISON"
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"

    @property
    def root_dir(self) -> Path:
        base_path = Path(__file__).parent.parent.parent
        if self is DomainToolkit.TABLE_COMPARISON:
            return base_path / "table_comparison"
        elif self is DomainToolkit.BINARY_CLASSIFICATION:
            return base_path / "binary_classification"
        else:
            raise ValueError(f"Unrecognized domain toolkit: {self}")

    @property
    def base_config_filepath(self) -> Path:
        return self.root_dir / "_config" / "raw.json"

    @property
    def config_override_filename(self) -> str:
        if self is DomainToolkit.TABLE_COMPARISON:
            return "table_comparison.json"
        elif self is DomainToolkit.BINARY_CLASSIFICATION:
            return "binary_classification.json"
        else:
            raise ValueError(f"Unrecognized domain toolkit: {self}")
