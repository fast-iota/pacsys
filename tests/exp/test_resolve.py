import pytest

import pacsys
from pacsys.device import Device
from pacsys.exp._resolve import resolve_backend, resolve_drf
from pacsys.testing import FakeBackend


class TestResolveDrf:
    def test_string_passthrough(self):
        assert resolve_drf("M:OUTTMP@p,1000") == "M:OUTTMP@p,1000"

    def test_device_object(self):
        dev = Device("M:OUTTMP").with_event("p,1000")
        assert "M:OUTTMP" in resolve_drf(dev)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected str or Device"):
            resolve_drf(42)


class TestResolveBackend:
    def test_explicit_backend(self):
        fake = FakeBackend()
        assert resolve_backend(fake) is fake

    def test_global_backend(self, monkeypatch):
        fake = FakeBackend()
        monkeypatch.setattr(pacsys, "_get_global_backend", lambda: fake)
        assert resolve_backend() is fake
