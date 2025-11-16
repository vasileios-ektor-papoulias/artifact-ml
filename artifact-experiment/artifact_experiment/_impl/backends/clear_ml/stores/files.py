from typing import Dict, Optional, Type, TypeVar

from clearml.binding.artifacts import Artifact as ClearMLArtifact

from artifact_experiment._impl.backends.clear_ml.stores.store import ClearMLStore

ClearMLFileStoreT = TypeVar("ClearMLFileStoreT", bound="ClearMLFileStore")


class ClearMLFileStore(ClearMLStore[ClearMLArtifact]):
    def __init__(
        self,
        root_dir: str,
        dict_all_files: Dict[str, ClearMLArtifact],
    ):
        super().__init__(root_dir=root_dir)
        self.__dict_all_files = dict_all_files

    @classmethod
    def build(
        cls: Type[ClearMLFileStoreT],
        dict_all_files: Dict[str, ClearMLArtifact],
        root_dir: Optional[str] = None,
    ) -> ClearMLFileStoreT:
        if root_dir is None:
            root_dir = ""
        dict_artifact_files = {
            file_name: file
            for file_name, file in dict_all_files.items()
            if file_name.startswith(root_dir)
        }
        store = cls(root_dir=root_dir, dict_all_files=dict_artifact_files)
        return store

    @property
    def dict_all_files(self) -> Dict[str, ClearMLArtifact]:
        return self.__dict_all_files.copy()

    @property
    def n_entries(self) -> int:
        return len(self.__dict_all_files)

    def get_n_files(self, path: str) -> int:
        return len(self.get_history(path=path))

    def get_history(self, path: str) -> Dict[str, ClearMLArtifact]:
        path = self._prepend_root_dir(path=path, root_dir=self._root_dir)
        return self._get_file_history(dict_all_files=self.__dict_all_files, path=path)

    def _get_history(self, path: str) -> Dict[str, ClearMLArtifact]:
        return self._get_file_history(dict_all_files=self.__dict_all_files, path=path)

    @classmethod
    def _get_file_history(
        cls,
        dict_all_files: Dict[str, ClearMLArtifact],
        path: str,
    ) -> Dict[str, ClearMLArtifact]:
        dict_file_history = {name: file for name, file in dict_all_files.items() if path in name}
        return dict_file_history

    def _get(self, path: str) -> ClearMLArtifact:
        return self._get_file(dict_all_files=self.__dict_all_files, path=path)

    @classmethod
    def _get_file(
        cls,
        dict_all_files: Dict[str, ClearMLArtifact],
        path: str,
    ) -> ClearMLArtifact:
        try:
            return dict_all_files[path]
        except KeyError:
            raise KeyError(f"Filepath: {path} doesn't exist")
