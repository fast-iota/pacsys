import pytest
from pacsys.exp._resolve import resolve_drf, resolve_backend
from pacsys.device import Device
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
