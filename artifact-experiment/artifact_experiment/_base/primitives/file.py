from dataclasses import dataclass


@dataclass(frozen=True)
class File:
    path_source: str
