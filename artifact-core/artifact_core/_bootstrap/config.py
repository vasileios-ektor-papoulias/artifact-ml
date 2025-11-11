import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Type, TypeVar

from artifact_core._bootstrap.libs.config_merger import ConfigMerger
from artifact_core._bootstrap.libs.override_locator import ConfigOverrideLocator
from artifact_core._bootstrap.toolkit import DomainToolkit

ToolkitConfigT = TypeVar("ToolkitConfigT", bound="ToolkitConfig")


@dataclass(frozen=True)
class ToolkitConfig:
    native_artifact_path: Optional[Path]
    custom_artifact_path: Optional[Path]
    dict_scores_config: Mapping[str, Any]
    dict_arrays_config: Mapping[str, Any]
    dict_plots_config: Mapping[str, Any]
    dict_score_collections_config: Mapping[str, Any]
    dict_array_collections_config: Mapping[str, Any]
    dict_plot_collections_config: Mapping[str, Any]
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
        user_override_dir = ConfigOverrideLocator.get_config_override_dir()
        user_override = ConfigOverrideLocator.get_config_override(domain_toolkit=domain_toolkit)
        merged = ConfigMerger.merge(base_config=dict(data), override=user_override)
        native_artifact_path = cls._resolve_native_path(
            domain_toolkit_root=domain_toolkit_root,
            relative_path=merged.get(cls._native_artifact_path_key),
        )
        custom_artifact_path = cls._resolve_custom_path(
            user_override_dir=user_override_dir,
            relative_path=merged.get(cls._custom_artifact_path_key),
        )
        return cls(
            native_artifact_path=native_artifact_path,
            custom_artifact_path=custom_artifact_path,
            dict_scores_config=merged.get(cls._scores_key, {}) or {},
            dict_arrays_config=merged.get(cls._arrays_key, {}) or {},
            dict_plots_config=merged.get(cls._plots_key, {}) or {},
            dict_score_collections_config=merged.get(cls._score_collections_key, {}) or {},
            dict_array_collections_config=merged.get(cls._array_collections_key, {}) or {},
            dict_plot_collections_config=merged.get(cls._plot_collections_key, {}) or {},
        )

    @staticmethod
    def _resolve_native_path(
        domain_toolkit_root: Path, relative_path: Optional[str]
    ) -> Optional[Path]:
        if relative_path:
            return (domain_toolkit_root / Path(relative_path)).resolve()

    @staticmethod
    def _resolve_custom_path(
        user_override_dir: Optional[Path], relative_path: Optional[str]
    ) -> Optional[Path]:
        if relative_path and user_override_dir:
            return (user_override_dir.parent / Path(relative_path)).resolve()
