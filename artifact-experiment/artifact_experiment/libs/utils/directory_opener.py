import subprocess
import sys


class DirectoryOpener:
    def open_directory(self, path: str):
        if self._is_mac():
            self._open_on_mac(path)
        elif self._is_windows():
            self._open_on_windows(path)
        elif self._is_linux():
            self._open_on_linux(path)
        else:
            self._fallback_print(path)

    def _is_windows(self) -> bool:
        return sys.platform.startswith("win")

    def _is_mac(self) -> bool:
        return sys.platform.startswith("darwin")

    def _is_linux(self) -> bool:
        return sys.platform.startswith("linux")

    def _open_on_windows(self, path: str):
        try:
            subprocess.Popen(
                f'start "" "{path}"',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self._fallback_print(path)

    def _open_on_mac(self, path: str):
        try:
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self._fallback_print(path)

    def _open_on_linux(self, path: str):
        try:
            subprocess.Popen(
                ["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            self._fallback_print(path)

    def _fallback_print(self, path: str):
        print(f"Directory is available at: {path}")
