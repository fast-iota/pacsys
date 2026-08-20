"""Fresh-process tests for startup import boundaries."""

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[1]


def _run_isolated(code: str) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_ROOT), existing)))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_scalar_dpm_path_does_not_import_numpy():
    _run_isolated(
        """
import sys

import pacsys
from pacsys.backends.dpm_http import _reply_to_value_and_type, _value_to_setting
from pacsys.dpm_protocol import Scalar_reply
from pacsys.types import Reading, ValueType

backend = pacsys.dpm(host="127.0.0.1", port=1, timeout=0.01)
backend.close()
reply = Scalar_reply()
reply.data = 72.5
assert _reply_to_value_and_type(reply) == (72.5, ValueType.SCALAR)
raw, scaled, text = _value_to_setting(1, 72.5)
assert raw is None and scaled.data == [72.5] and text is None
left = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)
right = Reading(drf="M:OUTTMP", value_type=ValueType.SCALAR, value=72.5)
assert left == right
assert hash(left) == hash(right)
assert left.to_dict()["value"] == 72.5
assert Reading.from_dict(left.to_dict()) == left
assert "numpy" not in sys.modules
assert "asyncio" not in sys.modules
"""
    )


def test_array_reply_imports_numpy_on_demand():
    _run_isolated(
        """
import sys

from pacsys.backends.dpm_http import _reply_to_value_and_type
from pacsys.dpm_protocol import ScalarArray_reply
from pacsys.types import ValueType

assert "numpy" not in sys.modules
reply = ScalarArray_reply()
reply.data = [1.0, 2.0]
value, value_type = _reply_to_value_and_type(reply)
assert value.tolist() == [1.0, 2.0]
assert value_type == ValueType.SCALAR_ARRAY
assert "numpy" in sys.modules
"""
    )


def test_acnet_facade_and_dir_do_not_load_subsystems():
    _run_isolated(
        """
import sys

import pacsys

acnet = pacsys.acnet
assert len(acnet.__all__) == len(set(acnet.__all__))
assert set(acnet.__all__) == set(acnet._LAZY_IMPORTS)
before = set(sys.modules)
assert set(acnet.__all__).issubset(dir(acnet))
assert {"errors", "ftp", "rad50"}.issubset(dir(acnet))
assert {"__all__", "__doc__", "__name__"}.issubset(dir(acnet))
assert set(sys.modules) == before
assert "asyncio" not in sys.modules
assert "pacsys.acnet.async_connection" not in sys.modules
assert "pacsys.acnet.ftp" not in sys.modules
assert "pacsys.acnet.errors" not in sys.modules
assert acnet.ACNET_PORT == 6801
assert "pacsys.acnet.constants" in sys.modules
assert "asyncio" not in sys.modules
"""
    )


def test_dpm_streaming_imports_asyncio_on_demand():
    _run_isolated(
        """
import sys

from pacsys.backends.dpm_http import DPMHTTPBackend

assert "asyncio" not in sys.modules
backend = DPMHTTPBackend(host="127.0.0.1", port=1, timeout=0.01)
assert "asyncio" not in sys.modules
backend._ensure_reactor()
assert "asyncio" in sys.modules
backend.close()
"""
    )


def test_acnet_errors_and_public_exports_resolve_lazily():
    _run_isolated(
        """
import sys

from pacsys.acnet.errors import ERR_OK

assert ERR_OK == 0
assert "asyncio" not in sys.modules
assert "pacsys.acnet.async_connection" not in sys.modules
assert "pacsys.acnet.ftp" not in sys.modules

import pacsys.acnet as acnet

for name in acnet.__all__:
    getattr(acnet, name)
assert acnet.rad50.encode("ABC") == 1683
"""
    )
