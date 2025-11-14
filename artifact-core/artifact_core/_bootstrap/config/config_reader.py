import json
from pathlib import Path
from typing import Any, Mapping, Optional

from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit
from artifact_core._utils.collections.map_merger import MapMerger


class ToolkitConfigReader:
    @classmethod
    def read(
        cls, domain_toolkit: DomainToolkit, override_dir: Optional[Path] = None
    ) -> Mapping[str, Any]:
        config = cls._read_base(domain_toolkit=domain_toolkit)
        if override_dir is not None:
            override = cls._read_override(domain_toolkit=domain_toolkit, override_dir=override_dir)
            if override is not None:
                config = cls._apply_override(base=config, override=override)
        return config

    @classmethod
    def _read_base(cls, domain_toolkit: DomainToolkit) -> Mapping[str, Any]:
        base_filepath = domain_toolkit.base_config_filepath
        return cls._read_json(filepath=base_filepath)

    @classmethod
    def _read_override(
        cls, domain_toolkit: DomainToolkit, override_dir: Path
    ) -> Optional[Mapping[str, Any]]:
        override_filepath = override_dir / domain_toolkit.config_override_filename
        if override_filepath.exists():
            return cls._read_json(filepath=override_filepath)

    @staticmethod
    def _read_json(filepath: Path) -> Mapping[str, Any]:
        with Path(filepath).open("r", encoding="utf-8") as f:
            strucutred_data = json.load(f)
        return strucutred_data

    @staticmethod
    def _apply_override(base: Mapping[str, Any], override: Mapping[str, Any]) -> Mapping[str, Any]:
        return MapMerger.merge(base=base, override=override)
