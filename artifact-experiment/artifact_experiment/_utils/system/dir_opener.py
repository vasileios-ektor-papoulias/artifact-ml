import os
import subprocess
import sys


class DirectoryOpener:
    @classmethod
    def open_directory(cls, path: str):
        if cls._is_mac():
            cls._open_on_mac(path)
        elif cls._is_windows():
            cls._open_on_windows(path)
        elif cls._is_linux():
            cls._open_on_linux(path)
        else:
            cls._fallback_print(path)

    @staticmethod
    def _is_windows() -> bool:
        return sys.platform.startswith("win")

    @staticmethod
    def _is_mac() -> bool:
        return sys.platform.startswith("darwin")

    @staticmethod
    def _is_linux() -> bool:
        return sys.platform.startswith("linux")

    @classmethod
    def _open_on_windows(cls, path: str):
        try:
            subprocess.Popen(
                f'start "" "{path}"',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            cls._fallback_print(path)

    @classmethod
    def _open_on_mac(cls, path: str):
        try:
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            cls._fallback_print(path)

    @classmethod
    def _open_on_linux(cls, path: str):
        try:
            subprocess.Popen(
                ["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            cls._fallback_print(path)

    @staticmethod
    def _fallback_print(path: str):
        if os.path.isdir(path):
            print(f"Directory is available at: {path}")
        else:
            print(f"Directory does not exist: {path}")
