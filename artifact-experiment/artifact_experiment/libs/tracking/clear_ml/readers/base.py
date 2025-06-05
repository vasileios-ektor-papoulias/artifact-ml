import os

from artifact_experiment.libs.tracking.clear_ml.config import ARTIFACT_ML_ROOT_DIR


class ClearMLReader:
    _root_dir = ARTIFACT_ML_ROOT_DIR

    @classmethod
    def _get_full_path(cls, path: str) -> str:
        return os.path.join(cls._root_dir, path)
