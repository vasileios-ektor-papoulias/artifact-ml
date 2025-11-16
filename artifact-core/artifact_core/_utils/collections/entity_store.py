from typing import (
    Dict,
    Generic,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Optional,
    TypeVar,
    ValuesView,
)

IdentifierType = Hashable

StoreDataT = TypeVar("StoreDataT")


class EntityStore(Generic[StoreDataT]):
    def __init__(self, initial: Optional[Mapping[IdentifierType, StoreDataT]] = None):
        self._data: Dict[IdentifierType, StoreDataT] = (
            {identifier: value for identifier, value in initial.items()}
            if initial is not None
            else {}
        )

    @property
    def keys(self) -> KeysView[IdentifierType]:
        return self._data.keys()

    @property
    def values(self) -> ValuesView[StoreDataT]:
        return self._data.values()

    @property
    def ids(self) -> Iterable[IdentifierType]:
        return list(self.keys)

    @property
    def n_items(self) -> int:
        return len(self._data)

    @property
    def items(self) -> ItemsView[IdentifierType, StoreDataT]:
        return self._data.items()

    @property
    def to_dict(self) -> Dict[IdentifierType, StoreDataT]:
        return self._data.copy()

    def __len__(self) -> int:
        return self.n_items

    def __contains__(self, identifier: IdentifierType) -> bool:
        return identifier in self._data

    def __iter__(self) -> Iterator[IdentifierType]:
        return iter(self._data)

    def get(self, identifier: IdentifierType) -> StoreDataT:
        try:
            return self._data[identifier]
        except KeyError:
            raise KeyError(f"Unknown identifier: {identifier!r}. Known: {list(self._data.keys())}")

    def set(self, identifier: IdentifierType, value: StoreDataT) -> None:
        self._data[identifier] = value

    def set_multiple(self, mapping: Mapping[IdentifierType, StoreDataT]) -> None:
        for identifier, value in mapping.items():
            self.set(identifier=identifier, value=value)

    def delete(self, identifier: IdentifierType) -> None:
        try:
            del self._data[identifier]
        except KeyError:
            raise KeyError(f"Unknown identifier: {identifier!r}. Known: {list(self._data.keys())}")

    def clear(self) -> None:
        self._data.clear()
