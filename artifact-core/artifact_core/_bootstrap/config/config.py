import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Type, TypeVar

from artifact_core._bootstrap.config.overrider import ConfigOverrider
from artifact_core._bootstrap.types.toolkit import DomainToolkit
from artifact_core._bootstrap.utils.mapping_merger import MappingMerger

ToolkitConfigT = TypeVar("ToolkitConfigT", bound="DomainToolkitConfig")


@dataclass(frozen=True)
class DomainToolkitConfig:
    native_artifact_path: Path
    custom_artifact_path: Optional[Path]
    scores_config: Mapping[str, Mapping[str, Any]]
    arrays_config: Mapping[str, Mapping[str, Any]]
    plots_config: Mapping[str, Mapping[str, Any]]
    score_collections_config: Mapping[str, Mapping[str, Any]]
    array_collections_config: Mapping[str, Mapping[str, Any]]
    plot_collections_config: Mapping[str, Mapping[str, Any]]
    _native_artifact_path_key: ClassVar[str] = "native_artifact_path"
    _custom_artifact_path_key: ClassVar[str] = "custom_artifact_path"
    _scores_key: ClassVar[str] = "scores"
    _arrays_key: ClassVar[str] = "arrays"
    _plots_key: ClassVar[str] = "plots"
    _score_collections_key: ClassVar[str] = "score_collections"
    _array_collections_key: ClassVar[str] = "array_collections"
    _plot_collections_key: ClassVar[str] = "plot_collections"

    @classmethod
    def from_json_file(
        cls: Type[ToolkitConfigT],
        filepath: Path,
        domain_toolkit: DomainToolkit,
        domain_toolkit_root: Path,
    ) -> ToolkitConfigT:
        with Path(filepath).open("r", encoding="utf-8") as f:
            base = json.load(f)
        return cls.from_json(
            data=base,
            domain_toolkit_root=domain_toolkit_root,
            domain_toolkit=domain_toolkit,
        )

    @classmethod
    def from_json(
        cls: Type[ToolkitConfigT],
        data: Mapping[str, Any],
        domain_toolkit_root: Path,
        domain_toolkit: DomainToolkit,
    ) -> ToolkitConfigT:
        user_override_dir = ConfigOverrider.get_config_override_dir()
        user_override = ConfigOverrider.get_config_override(domain_toolkit=domain_toolkit)
        merged = MappingMerger.merge(base=dict(data), override=user_override)
        native_artifact_path = cls._resolve_native_path(
            domain_toolkit_root=domain_toolkit_root,
            relative_path=cls._extract_native_path(config=merged),
        )
        custom_artifact_path = cls._resolve_custom_path(
            user_override_dir=user_override_dir,
            relative_path=cls._extract_custom_path(config=merged),
        )
        return cls(
            native_artifact_path=native_artifact_path,
            custom_artifact_path=custom_artifact_path,
            scores_config=merged.get(cls._scores_key, {}) or {},
            arrays_config=merged.get(cls._arrays_key, {}) or {},
            plots_config=merged.get(cls._plots_key, {}) or {},
            score_collections_config=merged.get(cls._score_collections_key, {}) or {},
            array_collections_config=merged.get(cls._array_collections_key, {}) or {},
            plot_collections_config=merged.get(cls._plot_collections_key, {}) or {},
        )

    @staticmethod
    def _resolve_native_path(domain_toolkit_root: Path, relative_path: Path) -> Path:
        return (domain_toolkit_root / relative_path).resolve()

    @staticmethod
    def _resolve_custom_path(
        user_override_dir: Optional[Path], relative_path: Optional[Path]
    ) -> Optional[Path]:
        if relative_path and user_override_dir:
            return (user_override_dir.parent / relative_path).resolve()

    @classmethod
    def _extract_native_path(cls, config: Dict[str, Any]) -> Path:
        native_path = config.get(cls._native_artifact_path_key)
        if native_path is None:
            raise ValueError("Missing native artifact path, likely due to invalid config override.")
        return Path(native_path)

    @classmethod
    def _extract_custom_path(cls, config: Dict[str, Any]) -> Optional[Path]:
        custom_path = config.get(cls._custom_artifact_path_key)
        if custom_path is not None:
            return Path(custom_path)
