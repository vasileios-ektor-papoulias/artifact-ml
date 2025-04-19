from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ArtifactResult,
    DataSpecProtocol,
    NoArtifactHyperparams,
)

dataSpecProtocolT = TypeVar("dataSpecProtocolT", bound="DataSpecProtocol")
artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound="ArtifactHyperparams")
artifactResourcesT = TypeVar("artifactResourcesT", bound="ArtifactResources")
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)
artifactT = TypeVar("artifactT", bound=Artifact)
artifactTypeT = TypeVar("artifactTypeT", bound="ArtifactType")


class ArtifactType(Enum):
    pass


class ArtifactRegistry(
    Generic[artifactTypeT, artifactResourcesT, artifactResultT, dataSpecProtocolT]
):
    _artifact_registry: Dict[
        artifactTypeT,
        Type[
            Artifact[
                artifactResourcesT,
                artifactResultT,
                ArtifactHyperparams,
                dataSpecProtocolT,
            ]
        ],
    ] = {}
    _artifact_config_registry: Dict[artifactTypeT, Type[ArtifactHyperparams]] = {}

    @classmethod
    @abstractmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._artifact_registry = {}
        cls._artifact_config_registry = {}

    @classmethod
    def register_artifact(
        cls, artifact_type: artifactTypeT
    ) -> Callable[[Type[artifactT]], Type[artifactT]]:
        def artifact_registration_decorator(
            subclass: Type[artifactT],
        ) -> Type[artifactT]:
            cls._artifact_registry[artifact_type] = subclass
            return subclass

        return artifact_registration_decorator

    @classmethod
    def register_artifact_config(
        cls, artifact_type: artifactTypeT
    ) -> Callable[[Type[artifactHyperparamsT]], Type[artifactHyperparamsT]]:
        def artifact_config_registration_decorator(
            subclass: Type[artifactHyperparamsT],
        ) -> Type[artifactHyperparamsT]:
            cls._artifact_config_registry[artifact_type] = subclass
            return subclass

        return artifact_config_registration_decorator

    @classmethod
    def get(
        cls, artifact_type: artifactTypeT, data_spec: dataSpecProtocolT
    ) -> Artifact[
        artifactResourcesT,
        artifactResultT,
        ArtifactHyperparams,
        dataSpecProtocolT,
    ]:
        artifact_class = cls._get_artifact_class(artifact_type=artifact_type)
        artifact_hyperparams_class = cls._get_artifact_hyperparams_class(
            artifact_type=artifact_type
        )
        artifact_config = cls._read_artifact_config(artifact_type=artifact_type)
        hyperparams = cls._build_artifact_hyperparams(
            artifact_hyperparams_class=artifact_hyperparams_class, artifact_config=artifact_config
        )
        artifact = artifact_class(data_spec=data_spec, hyperparams=hyperparams)
        return artifact

    @classmethod
    def _get_artifact_class(
        cls, artifact_type: artifactTypeT
    ) -> Type[
        Artifact[
            artifactResourcesT,
            artifactResultT,
            ArtifactHyperparams,
            dataSpecProtocolT,
        ]
    ]:
        artifact_class = cls._artifact_registry.get(artifact_type, None)
        if artifact_class is None:
            raise ValueError(f"Artifact {artifact_type} not registered")
        return artifact_class

    @classmethod
    def _build_artifact_hyperparams(
        cls,
        artifact_hyperparams_class: Optional[Type[ArtifactHyperparams]],
        artifact_config: Optional[Dict[str, Any]],
    ) -> Optional[ArtifactHyperparams]:
        if artifact_hyperparams_class is None:
            artifact_hyperparams = None
        elif artifact_hyperparams_class == NoArtifactHyperparams:
            artifact_hyperparams = None
        elif artifact_config is None:
            raise ValueError(
                f"Missing config for hyperparams type {artifact_hyperparams_class.__name__}"
            )
        else:
            try:
                artifact_hyperparams = artifact_hyperparams_class.build(**artifact_config)
            except TypeError as err:
                raise ValueError(
                    f"Error instantiating '{artifact_hyperparams_class.__name__}'"
                    f"with argumentss {artifact_config}: {err}"
                ) from err
        return artifact_hyperparams

    @classmethod
    def _get_artifact_hyperparams_class(
        cls, artifact_type: artifactTypeT
    ) -> Optional[Type[ArtifactHyperparams]]:
        artifact_hyperparams_class = cls._artifact_config_registry.get(artifact_type, None)
        return artifact_hyperparams_class

    @classmethod
    def _read_artifact_config(cls, artifact_type: artifactTypeT) -> Optional[Dict[str, Any]]:
        dict_artifact_configs = cls._get_artifact_configurations()
        artifact_config = dict_artifact_configs.get(artifact_type.name, None)
        return artifact_config
