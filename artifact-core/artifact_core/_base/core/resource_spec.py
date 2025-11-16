from typing import Protocol


class ResourceSpecProtocol(Protocol):
    pass


class NoResourceSpec(ResourceSpecProtocol):
    pass


NO_RESOURCE_SPEC = NoResourceSpec()
