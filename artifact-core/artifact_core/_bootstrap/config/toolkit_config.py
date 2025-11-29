from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Type, TypeVar

from artifact_core._bootstrap.config.config_reader import ToolkitConfigReader
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit

ToolkitConfigT = TypeVar("ToolkitConfigT", bound="ToolkitConfig")


@dataclass(frozen=True)
class ToolkitConfig:
    custom_artifacts_dir: Optional[Path]
    scores_config: Mapping[str, Mapping[str, Any]]
    arrays_config: Mapping[str, Mapping[str, Any]]
    plots_config: Mapping[str, Mapping[str, Any]]
    score_collections_config: Mapping[str, Mapping[str, Any]]
    array_collections_config: Mapping[str, Mapping[str, Any]]
    plot_collections_config: Mapping[str, Mapping[str, Any]]
    _custom_artifacts_dir_key: ClassVar[str] = "custom_artifacts_dir"
    _scores_key: ClassVar[str] = "scores"
    _arrays_key: ClassVar[str] = "arrays"
    _plots_key: ClassVar[str] = "plots"
    _score_collections_key: ClassVar[str] = "score_collections"
    _array_collections_key: ClassVar[str] = "array_collections"
    _plot_collections_key: ClassVar[str] = "plot_collections"

    @classmethod
    def load(
        cls: Type[ToolkitConfigT],
        domain_toolkit: DomainToolkit,
        config_override_dir: Optional[Path] = None,
    ) -> ToolkitConfigT:
        config = ToolkitConfigReader.read(
            domain_toolkit=domain_toolkit, override_dir=config_override_dir
        )
        return cls(
            custom_artifacts_dir=cls._extract_custom_artifacts_dir(config=config),
            scores_config=config.get(cls._scores_key, {}),
            arrays_config=config.get(cls._arrays_key, {}),
            plots_config=config.get(cls._plots_key, {}),
            score_collections_config=config.get(cls._score_collections_key, {}),
            array_collections_config=config.get(cls._array_collections_key, {}),
            plot_collections_config=config.get(cls._plot_collections_key, {}),
        )

    @classmethod
    def _extract_custom_artifacts_dir(cls, config: Mapping[str, Any]) -> Optional[Path]:
        custom_path = config.get(cls._custom_artifacts_dir_key)
        if custom_path is not None:
            return Path(custom_path)
