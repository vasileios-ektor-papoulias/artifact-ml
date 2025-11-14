from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Type, TypeVar

from artifact_core._bootstrap.config.config_reader import ToolkitConfigReader
from artifact_core._bootstrap.config.override_locator import ConfigOverrideLocator
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit

ToolkitConfigT = TypeVar("ToolkitConfigT", bound="ToolkitConfig")


@dataclass(frozen=True)
class ToolkitConfig:
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
    def load(cls: Type[ToolkitConfigT], domain_toolkit: DomainToolkit) -> ToolkitConfigT:
        override_dir = ConfigOverrideLocator.find()
        config = ToolkitConfigReader.read(domain_toolkit=domain_toolkit, override_dir=override_dir)
        native_artifact_path = cls._resolve_native_path(
            domain_toolkit_root=domain_toolkit.root_dir,
            relative_path=cls._extract_native_path(config=config),
        )
        custom_artifact_path = cls._resolve_custom_path(
            user_override_dir=override_dir,
            relative_path=cls._extract_custom_path(config=config),
        )
        return cls(
            native_artifact_path=native_artifact_path,
            custom_artifact_path=custom_artifact_path,
            scores_config=config.get(cls._scores_key, {}) or {},
            arrays_config=config.get(cls._arrays_key, {}) or {},
            plots_config=config.get(cls._plots_key, {}) or {},
            score_collections_config=config.get(cls._score_collections_key, {}) or {},
            array_collections_config=config.get(cls._array_collections_key, {}) or {},
            plot_collections_config=config.get(cls._plot_collections_key, {}) or {},
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
    def _extract_native_path(cls, config: Mapping[str, Any]) -> Path:
        native_path = config.get(cls._native_artifact_path_key)
        if native_path is None:
            raise ValueError("Missing native artifact path, likely due to invalid config override.")
        return Path(native_path)

    @classmethod
    def _extract_custom_path(cls, config: Mapping[str, Any]) -> Optional[Path]:
        custom_path = config.get(cls._custom_artifact_path_key)
        if custom_path is not None:
            return Path(custom_path)
