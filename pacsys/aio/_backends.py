"""Async backend abstract base class."""

from abc import ABC, abstractmethod
from typing import Optional

from pacsys.types import (
    Value,
    Reading,
    WriteResult,
    BackendCapability,
    ReadingCallback,
    ErrorCallback,
)


class AsyncBackend(ABC):
    """Async counterpart of Backend. Same capabilities, all methods async."""

    _closed: bool = False

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapability: ...

    @abstractmethod
    async def read(self, drf: str, timeout: Optional[float] = None) -> Value: ...

    @abstractmethod
    async def get(self, drf: str, timeout: Optional[float] = None) -> Reading: ...

    @abstractmethod
    async def get_many(self, drfs: list[str], timeout: Optional[float] = None) -> list[Reading]: ...

    async def write(self, drf: str, value: Value, timeout: Optional[float] = None) -> WriteResult:
        raise NotImplementedError("This backend does not support writes")

    async def write_many(self, settings: list[tuple[str, Value]], timeout: Optional[float] = None) -> list[WriteResult]:
        raise NotImplementedError("This backend does not support writes")

    async def subscribe(
        self,
        drfs: list[str],
        callback: Optional[ReadingCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ):
        raise NotImplementedError("This backend does not support streaming")

    async def remove(self, handle) -> None:
        raise NotImplementedError("This backend does not support streaming")

    async def stop_streaming(self) -> None:
        raise NotImplementedError("This backend does not support streaming")

    @property
    def authenticated(self) -> bool:
        return False

    @property
    def principal(self) -> Optional[str]:
        return None

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
