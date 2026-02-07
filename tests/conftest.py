"""
Shared pytest fixtures for pacsys unit tests.

This module provides common fixtures used across multiple test files.
"""

import pytest
from unittest import mock
from datetime import datetime

from pacsys.types import Reading, ValueType, DeviceMeta
from tests.devices import (
    make_jwt_token,
    MockGSSAPIModule,
    MockSocketWithReplies,
)


@pytest.fixture
def sample_reading():
    """Create a sample Reading for tests."""
    return Reading(
        drf="M:OUTTMP",
        value_type=ValueType.SCALAR,
        tag=1,
        facility_code=0,
        error_code=0,
        value=72.5,
        message=None,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        cycle=1234,
        meta=DeviceMeta(
            device_index=12345,
            name="M:OUTTMP",
            description="Outdoor Temperature",
            units="degF",
        ),
    )


@pytest.fixture
def error_reading():
    """Create an error Reading for tests."""
    return Reading(
        drf="M:BADDEV",
        value_type=ValueType.SCALAR,
        tag=1,
        facility_code=0,
        error_code=-42,
        value=None,
        message="Device not found",
        timestamp=None,
        cycle=None,
        meta=None,
    )


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = mock.MagicMock()
    backend.read.return_value = 72.5
    backend.get.return_value = Reading(
        drf="M:OUTTMP",
        value_type=ValueType.SCALAR,
        tag=1,
        facility_code=0,
        error_code=0,
        value=72.5,
        message=None,
        timestamp=None,
        cycle=None,
        meta=None,
    )
    return backend


@pytest.fixture
def sample_jwt():
    """Sample JWT token for testing."""
    return make_jwt_token({"sub": "testuser@fnal.gov", "exp": 9999999999})


@pytest.fixture
def mock_gssapi():
    """Fixture that patches gssapi module with MockGSSAPIModule.

    Usage:
        def test_something(mock_gssapi):
            from pacsys.auth import KerberosAuth
            auth = KerberosAuth()  # Uses mock gssapi
    """
    mock_module = MockGSSAPIModule()
    with mock.patch.dict("sys.modules", {"gssapi": mock_module}):
        yield mock_module


@pytest.fixture
def mock_dpm_socket():
    """Fixture that provides a mock DPM socket factory.

    Usage:
        def test_something(mock_dpm_socket):
            replies = [make_start_list(), make_scalar_reply()]
            with mock_dpm_socket(replies):
                backend = DPMHTTPBackend()
                reading = backend.get("M:OUTTMP")
    """

    class _MockDPMSocket:
        def __init__(self):
            self.socket = None
            self._patch = None

        def __call__(self, replies, list_id=1):
            self.socket = MockSocketWithReplies(list_id=list_id, replies=replies)
            return self

        def __enter__(self):
            self._patch = mock.patch("socket.socket", return_value=self.socket)
            self._patch.__enter__()
            return self.socket

        def __exit__(self, *args):
            self._patch.__exit__(*args)

    return _MockDPMSocket()


class MockACLResponse:
    """Mock httpx.Response for ACL backend testing."""

    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "http://test"),
                response=self,
            )
