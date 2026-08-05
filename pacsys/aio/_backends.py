"""Async backend abstract base class."""

from abc import ABC, abstractmethod
from typing import cast

from pacsys.errors import ReadError
from pacsys.types import (
    BackendCapability,
    ErrorCallback,
    Reading,
    ReadingCallback,
    Value,
    WriteResult,
)


class AsyncBackend(ABC):
    """Async counterpart of Backend. Same capabilities, all methods async."""

    _closed: bool = False

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapability: ...

    @abstractmethod
    async def read(self, drf: str, timeout: float | None = None) -> Value: ...

    @abstractmethod
    async def get(self, drf: str, timeout: float | None = None) -> Reading: ...

    @abstractmethod
    async def get_many(self, drfs: list[str], timeout: float | None = None) -> list[Reading]: ...

    async def read_many(self, drfs: list[str], timeout: float | None = None) -> list[Value]:
        """Read multiple device values in a single batch.

        Convenience wrapper around get_many() that extracts bare values
        and raises on any device error.
        """
        readings = await self.get_many(drfs, timeout=timeout)
        errors = [r for r in readings if not r.ok]
        if errors:
            failed = ", ".join(r.drf for r in errors)
            raise ReadError(readings, f"Device errors: {failed}")
        return [cast("Value", r.value) for r in readings]

    async def write(self, drf: str, value: Value, timeout: float | None = None) -> WriteResult:
        raise NotImplementedError("This backend does not support writes")

    async def write_many(self, settings: list[tuple[str, Value]], timeout: float | None = None) -> list[WriteResult]:
        raise NotImplementedError("This backend does not support writes")

    async def subscribe(
        self,
        drfs: list[str],
        callback: ReadingCallback | None = None,
        on_error: ErrorCallback | None = None,
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
    def principal(self) -> str | None:
        return None

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
