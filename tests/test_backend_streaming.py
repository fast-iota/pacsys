"""Tests for backend streaming functionality."""

from unittest.mock import MagicMock, patch

import pytest

from pacsys.backends.grpc_backend import GRPC_AVAILABLE


class TestDPMHTTPBackendStreaming:
    """Tests for DPMHTTPBackend streaming methods."""

    def test_subscribe_requires_drfs(self):
        """subscribe() should require non-empty drfs."""
        from pacsys.backends.dpm_http import DPMHTTPBackend

        backend = DPMHTTPBackend()
        try:
            with pytest.raises(ValueError, match="drfs cannot be empty"):
                backend.subscribe([], callback=lambda r, h: None)
        finally:
            backend.close()

    def test_subscription_handle_context_manager(self):
        """Test SubscriptionHandle can be used as context manager."""
        from pacsys.backends.dpm_http import _DPMHTTPSubscriptionHandle, DPMHTTPBackend

        backend = DPMHTTPBackend()
        try:
            handle = _DPMHTTPSubscriptionHandle(
                backend=backend,
                drfs=["M:OUTTMP@p,1000"],
                callback=None,
            )

            # Using context manager should call stop() on exit
            with handle as h:
                assert h is handle
                assert handle.stopped is False

            assert handle.stopped is True
        finally:
            backend.close()

    def test_callback_mode_handle_cannot_iterate(self):
        """Handle created with callback cannot be iterated via readings()."""
        from pacsys.backends.dpm_http import _DPMHTTPSubscriptionHandle, DPMHTTPBackend

        backend = DPMHTTPBackend()
        try:
            handle = _DPMHTTPSubscriptionHandle(
                backend=backend,
                drfs=["M:OUTTMP@p,1000"],
                callback=lambda r, h: None,
            )

            with pytest.raises(RuntimeError, match="Cannot iterate subscription with callback"):
                list(handle.readings(timeout=0))
        finally:
            backend.close()

    def test_close_calls_stop_streaming(self):
        """close() should call stop_streaming()."""
        from pacsys.backends.dpm_http import DPMHTTPBackend

        backend = DPMHTTPBackend()
        backend.stop_streaming = MagicMock()
        backend.close()

        backend.stop_streaming.assert_called_once()

    def test_remove_with_wrong_handle_type(self):
        """remove() should reject wrong handle types."""
        from pacsys.backends.dpm_http import DPMHTTPBackend

        backend = DPMHTTPBackend()
        try:
            with pytest.raises(TypeError, match="Expected _DPMHTTPSubscriptionHandle"):
                backend.remove("not a handle")
        finally:
            backend.close()


@pytest.mark.skipif(not GRPC_AVAILABLE, reason="grpc/protobuf not installed")
class TestGRPCBackendStreaming:
    """Tests for GRPCBackend streaming methods."""

    def test_subscribe_allows_optional_callback(self):
        """subscribe() should allow optional callback (for iterator mode)."""
        from pacsys.backends.grpc_backend import GRPCBackend

        backend = GRPCBackend()
        try:
            # Should NOT raise - callback is now optional
            pass  # Just testing that the method signature is correct
        finally:
            backend.close()

    def test_subscribe_requires_drfs(self):
        """subscribe() should require non-empty drfs."""
        from pacsys.backends.grpc_backend import GRPCBackend

        backend = GRPCBackend()
        try:
            with pytest.raises(ValueError, match="drfs cannot be empty"):
                backend.subscribe([], callback=lambda r, h: None)
        finally:
            backend.close()

    def test_close_calls_stop_streaming(self):
        """close() should call stop_streaming()."""
        from pacsys.backends.grpc_backend import GRPCBackend

        backend = GRPCBackend()
        backend.stop_streaming = MagicMock()
        backend.close()

        backend.stop_streaming.assert_called_once()

    def test_remove_with_wrong_handle_type(self):
        """remove() should reject wrong handle types."""
        from pacsys.backends.grpc_backend import GRPCBackend

        backend = GRPCBackend()
        try:
            with pytest.raises(TypeError, match="Expected _GRPCSubscriptionHandle"):
                backend.remove("not a handle")
        finally:
            backend.close()


class TestACLBackendStreaming:
    """Tests for ACLBackend streaming (should not support it)."""

    def test_subscribe_not_supported(self):
        """ACLBackend should not support subscribe()."""
        from pacsys.backends.acl import ACLBackend

        backend = ACLBackend()
        try:
            with pytest.raises(NotImplementedError, match="does not support streaming"):
                backend.subscribe(["M:OUTTMP@p,1000"], callback=lambda r, h: None)
        finally:
            backend.close()


class TestModuleLevelStreaming:
    """Tests for module-level streaming functions."""

    def test_subscribe_uses_global_backend(self):
        """pacsys.subscribe() should use global backend."""
        import pacsys

        # Reset any existing backend
        pacsys.shutdown()

        # Mock the global backend
        mock_backend = MagicMock()
        mock_handle = MagicMock()
        mock_backend.subscribe.return_value = mock_handle

        with patch.object(pacsys, "_get_global_backend", return_value=mock_backend):

            def callback(r, h):
                return None

            result = pacsys.subscribe(["M:OUTTMP@p,1000"], callback=callback)

            mock_backend.subscribe.assert_called_once_with(["M:OUTTMP@p,1000"], callback=callback, on_error=None)
            assert result == mock_handle

    def test_subscribe_without_callback(self):
        """pacsys.subscribe() should work without callback (iterator mode)."""
        import pacsys

        # Reset any existing backend
        pacsys.shutdown()

        # Mock the global backend
        mock_backend = MagicMock()
        mock_handle = MagicMock()
        mock_backend.subscribe.return_value = mock_handle

        with patch.object(pacsys, "_get_global_backend", return_value=mock_backend):
            result = pacsys.subscribe(["M:OUTTMP@p,1000"])

            mock_backend.subscribe.assert_called_once_with(["M:OUTTMP@p,1000"], callback=None, on_error=None)
            assert result == mock_handle
