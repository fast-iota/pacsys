#!/usr/bin/env python3
"""Validate Python scaling against Java reference implementation.

Generates test vectors, runs both implementations with identical inputs,
and compares outputs to detect port divergences.

Usage:
    cd tests/scaling_validation
    javac ScalingHarness.java
    PYTHONPATH=/mnt/pacsys python validate_scaling.py
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
from pathlib import Path

# Add pacsys to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pacsys.scaling import (
    _common_scale,
    _common_unscale,
    _primary_scale,
    _primary_unscale,
)

JAVA_DIR = Path(__file__).parent
REL_TOL = 1e-5  # relative tolerance for double comparisons
REL_TOL_FLOAT = 5e-4  # relaxed tolerance for cases with Java (float) casts
ABS_TOL = 1e-10  # absolute tolerance for near-zero comparisons


def float_bits(f: float) -> int:
    """Convert float to its IEEE 754 32-bit signed int representation."""
    return struct.unpack(">i", struct.pack(">f", f))[0]


# ---- Test vector generation ----


def _valid_input_lens(p_index: int) -> list[int]:
    """Return valid input_lens for a given primary transform index."""
    if p_index in (54, 56, 66, 72, 74, 82):
        return [2]
    if p_index in (76, 84):
        return [4]
    if p_index in (22, 24, 28):
        return [4]  # word-swap transforms need 4-byte input
    if p_index in (16, 46, 48, 50, 60, 78, 80):
        return [4]  # IEEE float transforms
    return [1, 2, 4]


def _raw_values_for(p_index: int, input_len: int) -> list[int]:
    """Representative raw values for a (p_index, input_len) pair."""
    # IEEE float transforms: use float-encoded raw values
    if p_index in (16, 48, 50, 60, 78, 80):
        floats = [0.0, 1.0, -1.0, 3.14, 0.036, 0.002, 5.0, 7.0, 10.0, -10.0]
        if p_index == 78:
            floats = [0.0, 1.0, 2.5, 3.0, 5.0]  # clamped 0-5
        if p_index == 80:
            floats = [0.0, 1.0, 5.0, 7.0, 10.0]  # clamped 0-10
        if p_index == 50:
            floats = [0.0, 1.0, -1.0, 5.0, 10.0, -10.0]  # clamped ±10.24
        return [float_bits(f) for f in floats]

    # p84: byte-reversed IEEE float
    if p_index == 84:
        vals = [1.0, 1.5, 3.14, -2.0, 0.0]
        result = []
        for v in vals:
            bits = float_bits(v)
            b = struct.pack(">i", bits)
            rev = int.from_bytes(b[::-1], "big")
            # Make signed
            if rev >= 0x80000000:
                rev -= 0x100000000
            result.append(rev)
        return result

    # p22, p24: word-swapped IEEE float input
    if p_index in (22, 24):
        vals = [1.0, 2.5, -3.0, 0.0]
        result = []
        for v in vals:
            if p_index == 22:
                bits = float_bits(v * 4.0)  # DEC float = IEEE * 4
            else:
                bits = float_bits(v)
            # Word-swap for input
            swapped = ((bits & 0xFFFF) << 16) | ((bits >> 16) & 0xFFFF)
            if swapped >= 0x80000000:
                swapped -= 0x100000000
            result.append(swapped)
        return result

    # p54: 0x0000-0x7FFF range (no bit 15)
    if p_index == 54:
        return [0, 1, 100, 1000, 16384, 32767]

    # p56, p72: unsigned 16-bit (raw used directly as unsigned)
    if p_index in (56, 72):
        return [0, 1, 16384, 32768, 50000, 65535]

    # p66: must be > 0 after sign-extend
    if p_index == 66:
        return [1, 100, 1000, 3200, 16384, 32767]

    # p82: 12-bit unsigned (top nibble must be 0)
    if p_index == 82:
        return [0, 1, 100, 1000, 2048, 4095]

    # p44: BCD values (32-bit)
    if p_index == 44:
        return [0x01234567, 0x00000001, 0x00099999, 0]

    # p46: unsigned 32-bit (include -1 which maps to unsigned 4294967295)
    if p_index == 46:
        return [0, 1, 1000, 2147483647, -1, -100]

    # p28: word-swap 32-bit
    if p_index == 28:
        return [0x00010002, 0x00000001, 0x7FFF0000, -1]

    # p58: unsigned (raw used directly)
    if p_index == 58:
        return [0, 1, 256, 1000, 65536]

    # p76: word-swap unsigned 32-bit
    if p_index == 76:
        return [0x00010000, 0, 0x00020001, -1]

    # Default based on input_len
    if input_len == 1:
        return [0, 1, 64, 127, 128, 200, 255]
    if input_len == 2:
        return [0, 1, 100, 1000, 3200, 16384, 32767, 32768, 50000, 65535]
    # input_len == 4
    return [0, 1, 1000, 100000, 2147483647, -1, -1000]


def generate_primary_scale_vectors() -> list[tuple[str, int, int, int]]:
    """Generate (line, p_index, input_len, raw) test vectors for primary scale."""
    vectors = []
    for p_index in range(0, 86, 2):
        if p_index == 68:
            # Always errors, include one test to verify
            vectors.append((f"PS,{p_index},2,0", p_index, 2, 0))
            continue
        for input_len in _valid_input_lens(p_index):
            for raw in _raw_values_for(p_index, input_len):
                line = f"PS,{p_index},{input_len},{raw}"
                vectors.append((line, p_index, input_len, raw))
    return vectors


def generate_primary_unscale_vectors(
    scale_results: list[tuple[int, int, float]],
) -> list[tuple[str, int, int, float]]:
    """Generate primary unscale vectors from successful scale results.

    Args:
        scale_results: list of (p_index, input_len, primary_value) from successful scales
    """
    vectors = []
    for p_index, input_len, primary_val in scale_results:
        if p_index == 68:
            continue
        line = f"PU,{p_index},{input_len},{primary_val}"
        vectors.append((line, p_index, input_len, primary_val))

    # Explicit boundary tests for overflow-sensitive transforms
    for p_index, input_len, val in [
        (64, 4, 1.0),  # boundary: Java (int)(1.0*2^31) saturates to MAX_INT
        (64, 4, -1.0),  # boundary: Java (int)(-1.0*2^31) = -2^31
    ]:
        line = f"PU,{p_index},{input_len},{val}"
        vectors.append((line, p_index, input_len, val))

    return vectors


# Common transform constant sets: (c_index, constants, [primary_values])
COMMON_TEST_CASES: list[tuple[int, tuple[float, ...], list[float]]] = [
    (0, (), [0.0, 1.0, -1.0, 42.0, 100.0]),
    (2, (100.0, 10.0, 5.0), [0.0, 1.0, 3.0, -1.0, 5.0]),
    (2, (1.0, 1.0, 0.0), [0.0, 1.0, 10.0]),
    (4, (10.0, 5.0), [0.0, 15.0, 20.0, 100.0]),
    (6, (3.0, 2.0), [0.0, 1.0, 4.0, 10.0]),
    (8, (1.0, 0.5, 2.0, 1.0), [1.0, 3.0, 5.0]),
    (10, (2.0, 10.0, 5.0), [1.0, 2.0, 5.0]),
    (12, (0.0, 0.0, 0.0, 2.0, 1.0), [0.0, 1.0, 3.0, 5.0]),
    (12, (0.0, 1.0, 0.0, 2.0, 1.0), [0.0, 1.0, 3.0]),
    (14, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), [0.0, 1.0, -1.0]),
    (14, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0), [0.0, 1.0, 2.0]),
    (16, (1.0, 1.0, 1.0, 1.0), [0.0, 1.0, 2.0, 5.0]),
    (18, (0.0, 0.0, 1.0, 0.0, 0.0, 1.0), [0.0, 1.0]),
    (20, (0.0, 1.0, 0.0), [10.0, 100.0, 1000.0]),
    (20, (1.0, 1.0, 0.0), [10.0, 100.0]),  # non-zero c[0] for invertibility
    (22, (1.0, 1.0), [0.0, 1.0, 2.0, -1.0]),
    (24, (5.0, 2.0, 1.0, 0.0, 0.1, 0.0), [1.0, 3.0, 10.0]),
    (26, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), [0.0, 1.0, 5.0]),
    (28, (1.0, 2.0, 10.0, 3.0), [1.0, 2.0, 5.0]),
    (30, (2.0, 0.0, 0.0, 1.0, 0.0, 99.0), [1.0, 3.0, 5.0]),
    (32, (1.0, 2.0, 0.0, 0.0), [1.0, 3.0, 5.0]),
    (32, (1.0, 1.0, 0.0, 0.0), [1.0, math.e, 10.0]),
    (34, (1.0, 0.0, 0.0, 1.0), [1.0, 3.0, 7.0]),
    (36, (0.0, 1.0, 0.0), [1.0, 4.0, 9.0, 16.0]),
    (36, (0.0, 2.0, 0.0), [1.0, 4.0, 9.0]),
    (38, (1.0, 1.0, 0.0, 0.0, 0.0, 100.0), [50.0]),
    (38, (1.0, -0.01, 0.0, 0.0, 0.0, 0.0), [1.0, 5.0, 10.0]),
    (40, (10.0, 2.0, 1.0, 0.0, 0.0, 0.0), [0.0, 1.0, 4.0]),
    (42, (5.0, 1.0, 0.0, 0.0, 0.1, 0.0), [1.0, 3.0, 10.0]),
    (44, (5.0, 1.0, 0.0, 2.0, 0.0), [1.0, 3.0, 10.0]),
    (46, (5.0, 1.0, 0.0, 0.0, 2.0, 0.0), [1.0, 3.0, 10.0]),
    (48, (1.0, 2.0, 1.0), [1.0, 2.0, 3.0]),
    (50, (1.0, 1.0), [0.0, 0.5, 0.9]),
    (52, (5.0, 0.0, 0.0, 0.0, 0.0), [1.0, 3.0, 10.0]),
    (54, (5.0, 0.0, 0.0, 0.0, 0.0, 0.0), [1.0, 3.0, 10.0]),
    (62, (1.0, 1.0, 0.0), [0.0, 1.0, 2.0]),
    (64, (0.0,), [0.0, 1.0, 2.0, 4.0]),  # N2 VPT
    (64, (1.0,), [0.0, 0.5, 0.8]),  # He VPT
    (66, (1.0, 1.0, 0.0, 0.0), [0.0, 1.0, 3.0]),
    (68, (1.0, 1.0, 0.0, 0.0, 1.0, 1.0), [math.e, 5.0, 10.0]),
    (70, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), [0.0, 1.0, 2.0]),
    (72, (1.0, 0.0, 1.0, 0.0, 0.0, 0.0), [1.0, 10.0, 100.0]),
    (74, (1.0, 2.0, 0.1, 1.0, 0.5, 0.1), [1.0, 2.0, 3.0]),
    (76, (5.0, 1.0, 2.0, 1.0, 0.1, 0.0), [1.0, 3.0, 10.0]),
    (76, (5.0, 1.0, 0.5, 1.0, 0.1, 0.0), [-2.0, -1.0, 0.0]),  # neg base + fractional exp
    (78, (1.0, 1.0, 0.0, 0.0), [0.0, 1.0, 2.0]),
    (80, (), [0.0, 1.0, 42.0, -1.0]),
    (82, (1.0, 1.0, 0.0, 1.0), [1.0, 10.0, 100.0]),  # c[3]=1 so log10(x+1) defined at x=0
    (86, (2.0, 8.0, -1.0, 10.0, -0.1, 1.0), [1.0, 5.0, 10.0]),
    (88, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), [0.0, 1.0, 5.0]),
    (88, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), [0.0, 1.0, 3.0]),
]


def _format_constants(c: tuple[float, ...]) -> str:
    """Format constants for CSV (colon-separated, or _ for empty)."""
    if not c:
        return "_"
    return ":".join(str(v) for v in c)


def generate_common_scale_vectors() -> list[tuple[str, int, tuple[float, ...], float]]:
    """Generate common scale test vectors."""
    vectors = []
    for c_index, constants, x_values in COMMON_TEST_CASES:
        cstr = _format_constants(constants)
        for x in x_values:
            line = f"CS,{c_index},{cstr},{x}"
            vectors.append((line, c_index, constants, x))

    # Add error cases
    for c_index in (56, 58, 60, 84, 90, 201):
        c = (1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cstr = _format_constants(c)
        line = f"CS,{c_index},{cstr},1.0"
        vectors.append((line, c_index, c, 1.0))

    return vectors


def _choose_p_index_for_unscale(c_index: int, primary_in: float) -> int:
    """Choose a p_index with search limits that bracket primary_in.

    Indices that use binary search need limits appropriate for the
    primary value range.  p_index=0 gives [-10.24, 10.235], good
    for small-range transforms.  p_index=10 gives huge range.
    p_index=8 gives [0, 65536] for positive values.
    """
    # Transforms with analytical inverse don't need limits
    analytical = {0, 2, 4, 6, 8, 10, 28, 32, 34, 36, 40, 50, 62, 66, 76, 78, 80, 86}
    if c_index in analytical:
        return 10  # doesn't matter, no binary search
    # For binary search transforms, pick limits that bracket primary_in
    if primary_in >= 0:
        return 8  # lowlim=0, uprlim=65536
    return 0  # lowlim=-10.24, uprlim=10.235


def generate_common_unscale_vectors(
    scale_results: list[tuple[int, tuple[float, ...], float, float]],
) -> list[tuple[str, int, tuple[float, ...], int, float]]:
    """Generate common unscale vectors from successful scale results.

    Args:
        scale_results: list of (c_index, constants, primary_in, common_out)
    """
    vectors = []
    # Skip c_indices that don't support unscale or have unsupported dependencies
    skip_unscale = {56, 58, 60, 84, 90, 201}
    for c_index, constants, primary_in, common_out in scale_results:
        if c_index in skip_unscale:
            continue
        if math.isnan(common_out) or math.isinf(common_out):
            continue
        # Skip degenerate inverse cases:
        # c20: inverse has c[0]^2 in denominator - needs c[0] != 0
        if c_index == 20 and len(constants) >= 1 and constants[0] == 0.0:
            continue
        # c48: 1/x singularity - binary search limits include x=0
        if c_index == 48:
            continue
        p_index = _choose_p_index_for_unscale(c_index, primary_in)
        cstr = _format_constants(constants)
        line = f"CU,{c_index},{cstr},{p_index},{common_out}"
        vectors.append((line, c_index, constants, p_index, common_out))
    return vectors


# ---- Run implementations ----


def compile_java() -> None:
    """Compile the Java harness."""
    print("Compiling ScalingHarness.java...")
    proc = subprocess.run(
        ["javac", "ScalingHarness.java"],
        cwd=str(JAVA_DIR),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"COMPILE ERROR:\n{proc.stderr}")
        sys.exit(1)
    print("  OK")


def run_java(lines: list[str]) -> list[str]:
    """Run Java harness with test vectors, return result lines."""
    input_text = "\n".join(lines) + "\n"
    proc = subprocess.run(
        ["java", "-cp", str(JAVA_DIR), "ScalingHarness"],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        print(f"JAVA ERROR:\n{proc.stderr}")
        sys.exit(1)
    results = proc.stdout.strip().split("\n")
    return results


def parse_result(line: str) -> tuple[bool, str]:
    """Parse a result line into (is_ok, value_or_error)."""
    if line.startswith("# skip"):
        return False, "skip"
    parts = line.split("\t", 1)
    if len(parts) != 2:
        return False, f"malformed: {line}"
    return parts[0] == "OK", parts[1]


def results_match(
    py_ok: bool,
    py_val: str,
    java_ok: bool,
    java_val: str,
    rel_tol: float = REL_TOL,
) -> tuple[bool, str]:
    """Compare Python and Java results. Returns (match, detail)."""
    if not py_ok and not java_ok:
        return True, "both error"
    if py_ok != java_ok:
        return False, f"py={'OK' if py_ok else 'ERR'}({py_val}) java={'OK' if java_ok else 'ERR'}({java_val})"

    # Both OK - compare numeric values
    try:
        pv = float(py_val)
        jv = float(java_val)
    except ValueError:
        return py_val == java_val, f"py={py_val} java={java_val}"

    # Handle NaN
    if math.isnan(pv) and math.isnan(jv):
        return True, "both NaN"
    if math.isnan(pv) or math.isnan(jv):
        return False, f"py={pv} java={jv}"

    # Handle Inf
    if math.isinf(pv) and math.isinf(jv):
        return pv == jv, f"py={pv} java={jv}"
    if math.isinf(pv) or math.isinf(jv):
        return False, f"py={pv} java={jv}"

    # Numeric comparison
    denom = max(abs(pv), abs(jv), 1e-10)
    rel_err = abs(pv - jv) / denom
    if rel_err <= rel_tol or abs(pv - jv) <= ABS_TOL:
        return True, f"py={pv} java={jv} rel_err={rel_err:.2e}"
    return False, f"py={pv} java={jv} rel_err={rel_err:.2e}"


def int_results_match(
    py_ok: bool,
    py_val: str,
    java_ok: bool,
    java_val: str,
) -> tuple[bool, str]:
    """Compare integer results (primary unscale). Allow ±1 for truncation differences."""
    if not py_ok and not java_ok:
        return True, "both error"
    if py_ok != java_ok:
        return False, f"py={'OK' if py_ok else 'ERR'}({py_val}) java={'OK' if java_ok else 'ERR'}({java_val})"
    try:
        pv = int(float(py_val))
        jv = int(float(java_val))
    except ValueError:
        return py_val == java_val, f"py={py_val} java={java_val}"
    if abs(pv - jv) <= 1:
        return True, f"py={pv} java={jv}"
    return False, f"py={pv} java={jv} diff={abs(pv - jv)}"


# ---- Main validation ----


# Common unscale cases with relaxed tolerance:
# - Java (float) casts reduce precision (c20, c32, c50)
# - Binary search convergence varies between implementations
RELAXED_UNSCALE = {20, 32, 50, 82, 14, 26, 88, 64, 68, 72}


def run_primary_scale_validation() -> tuple[int, int, list[str], list[tuple[int, int, float]]]:
    """Run primary scale validation. Returns (pass, fail, failures, successful_results)."""
    vectors = generate_primary_scale_vectors()
    lines = [v[0] for v in vectors]

    # Run Python
    py_results: list[tuple[bool, str]] = []
    for _line, p_index, input_len, raw in vectors:
        try:
            result = _primary_scale(raw, p_index, input_len)
            py_results.append((True, str(result)))
        except Exception as e:
            py_results.append((False, str(e)))

    # Run Java
    java_lines = run_java(lines)
    java_results = [parse_result(jl) for jl in java_lines]

    # Compare
    passed = 0
    failed = 0
    failures: list[str] = []
    successful: list[tuple[int, int, float]] = []

    for i, (line, p_index, input_len, raw) in enumerate(vectors):
        py_ok, py_val = py_results[i]
        java_ok, java_val = java_results[i]
        match, detail = results_match(py_ok, py_val, java_ok, java_val)
        if match:
            passed += 1
            if py_ok:
                successful.append((p_index, input_len, float(py_val)))
        else:
            failed += 1
            failures.append(f"  PS p{p_index} len={input_len} raw={raw}: {detail}")

    return passed, failed, failures, successful


def run_primary_unscale_validation(
    scale_results: list[tuple[int, int, float]],
) -> tuple[int, int, list[str]]:
    """Run primary unscale validation."""
    vectors = generate_primary_unscale_vectors(scale_results)
    if not vectors:
        return 0, 0, []

    lines = [v[0] for v in vectors]

    # Run Python
    py_results: list[tuple[bool, str]] = []
    for _line, p_index, input_len, value in vectors:
        try:
            result = _primary_unscale(value, p_index, input_len)
            py_results.append((True, str(result)))
        except Exception as e:
            py_results.append((False, str(e)))

    # Run Java
    java_lines = run_java(lines)
    java_results = [parse_result(jl) for jl in java_lines]

    # Compare
    passed = 0
    failed = 0
    failures: list[str] = []

    for i, (line, p_index, input_len, value) in enumerate(vectors):
        py_ok, py_val = py_results[i]
        java_ok, java_val = java_results[i]
        match, detail = int_results_match(py_ok, py_val, java_ok, java_val)
        if match:
            passed += 1
        else:
            failed += 1
            failures.append(f"  PU p{p_index} len={input_len} val={value}: {detail}")

    return passed, failed, failures


def run_common_scale_validation() -> tuple[int, int, list[str], list[tuple[int, tuple[float, ...], float, float]]]:
    """Run common scale validation."""
    vectors = generate_common_scale_vectors()
    lines = [v[0] for v in vectors]

    # Run Python
    py_results: list[tuple[bool, str]] = []
    for _line, c_index, constants, x in vectors:
        try:
            result = _common_scale(x, c_index, constants)
            py_results.append((True, str(result)))
        except Exception as e:
            py_results.append((False, str(e)))

    # Run Java
    java_lines = run_java(lines)
    java_results = [parse_result(jl) for jl in java_lines]

    # Compare
    passed = 0
    failed = 0
    failures: list[str] = []
    successful: list[tuple[int, tuple[float, ...], float, float]] = []

    for i, (line, c_index, constants, x) in enumerate(vectors):
        py_ok, py_val = py_results[i]
        java_ok, java_val = java_results[i]
        match, detail = results_match(py_ok, py_val, java_ok, java_val)
        if match:
            passed += 1
            if py_ok:
                successful.append((c_index, constants, x, float(py_val)))
        else:
            failed += 1
            failures.append(f"  CS c{c_index} consts={constants} x={x}: {detail}")

    return passed, failed, failures, successful


def run_common_unscale_validation(
    scale_results: list[tuple[int, tuple[float, ...], float, float]],
) -> tuple[int, int, list[str]]:
    """Run common unscale validation."""
    vectors = generate_common_unscale_vectors(scale_results)
    if not vectors:
        return 0, 0, []

    lines = [v[0] for v in vectors]

    # Run Python
    py_results: list[tuple[bool, str]] = []
    for _line, c_index, constants, p_index, xx in vectors:
        try:
            result = _common_unscale(xx, c_index, constants, p_index)
            py_results.append((True, str(result)))
        except Exception as e:
            py_results.append((False, str(e)))

    # Run Java
    java_lines = run_java(lines)
    java_results = [parse_result(jl) for jl in java_lines]

    # Compare
    passed = 0
    failed = 0
    failures: list[str] = []

    for i, (line, c_index, constants, p_index, xx) in enumerate(vectors):
        py_ok, py_val = py_results[i]
        java_ok, java_val = java_results[i]
        tol = REL_TOL_FLOAT if c_index in RELAXED_UNSCALE else REL_TOL
        match, detail = results_match(py_ok, py_val, java_ok, java_val, rel_tol=tol)
        if match:
            passed += 1
        else:
            failed += 1
            failures.append(f"  CU c{c_index} consts={constants} p={p_index} xx={xx}: {detail}")

    return passed, failed, failures


def main() -> None:
    compile_java()

    total_pass = 0
    total_fail = 0
    all_failures: list[str] = []

    # 1. Primary scale
    print("\n--- Primary Scale (raw -> primary) ---")
    ps_pass, ps_fail, ps_failures, ps_results = run_primary_scale_validation()
    total_pass += ps_pass
    total_fail += ps_fail
    all_failures.extend(ps_failures)
    print(f"  {ps_pass} passed, {ps_fail} failed")

    # 2. Primary unscale (round-trip from scale results)
    print("\n--- Primary Unscale (primary -> raw) ---")
    pu_pass, pu_fail, pu_failures = run_primary_unscale_validation(ps_results)
    total_pass += pu_pass
    total_fail += pu_fail
    all_failures.extend(pu_failures)
    print(f"  {pu_pass} passed, {pu_fail} failed")

    # 3. Common scale
    print("\n--- Common Scale (primary -> common) ---")
    cs_pass, cs_fail, cs_failures, cs_results = run_common_scale_validation()
    total_pass += cs_pass
    total_fail += cs_fail
    all_failures.extend(cs_failures)
    print(f"  {cs_pass} passed, {cs_fail} failed")

    # 4. Common unscale (round-trip from scale results)
    print("\n--- Common Unscale (common -> primary) ---")
    cu_pass, cu_fail, cu_failures = run_common_unscale_validation(cs_results)
    total_pass += cu_pass
    total_fail += cu_fail
    all_failures.extend(cu_failures)
    print(f"  {cu_pass} passed, {cu_fail} failed")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    if all_failures:
        print(f"\nFAILURES ({len(all_failures)}):")
        for f in all_failures:
            print(f)
    print(f"{'=' * 60}")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
