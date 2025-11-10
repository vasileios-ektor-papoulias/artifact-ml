import os
import shutil
import tempfile
from uuid import uuid4


class ManagedTempDir:
    def __init__(self):
        self._temp_dir = tempfile.mkdtemp(prefix="artifact_tracking_")

    @property
    def path(self) -> str:
        return self._temp_dir

    def copy_file(self, source_path: str) -> str:
        base_name = os.path.basename(source_path)
        name_without_ext, ext = os.path.splitext(base_name)
        unique_filename = f"{name_without_ext}_{uuid4()}{ext}"
        dest_path = os.path.join(self._temp_dir, unique_filename)
        shutil.copy2(source_path, dest_path)
        return dest_path

    def remove_file(self, path: str):
        if os.path.exists(path):
            os.remove(path)

    def cleanup(self):
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
