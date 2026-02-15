#!/usr/bin/env python3
"""Regenerate pacsys/_proto/ from interface-definitions submodule.

Usage:
    python scripts/generate_proto.py

Prerequisites:
    pip install grpcio-tools

This script:
  1. Compiles .proto files from interface-definitions/ using grpc_tools.protoc
  2. Writes generated *_pb2.py and *_pb2_grpc.py to pacsys/_proto/
  3. Creates __init__.py files for all intermediate packages
  4. Patches module names in generated code so imports work as pacsys._proto.*
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IFACE_DIR = ROOT / "interface-definitions"
PROTO_ROOT = IFACE_DIR / "proto"
OUTPUT_DIR = ROOT / "pacsys" / "_proto"

# Proto files we actually need, relative to IFACE_DIR (must match import paths in .proto files)
PROTO_FILES = [
    "proto/controls/common/v1/device.proto",
    "proto/controls/common/v1/status.proto",
    "proto/controls/service/DAQ/v1/DAQ.proto",
    "proto/controls/service/DevDB/v1/DevDB.proto",
]


def check_prerequisites():
    if not IFACE_DIR.exists():
        print(f"ERROR: {IFACE_DIR} not found. Run: git submodule update --init", file=sys.stderr)
        sys.exit(1)
    for pf in PROTO_FILES:
        full = IFACE_DIR / pf
        if not full.exists():
            print(f"ERROR: {full} not found. Is interface-definitions submodule up to date?", file=sys.stderr)
            sys.exit(1)
    try:
        from grpc_tools import protoc  # noqa: F401
    except ImportError:
        print("ERROR: grpcio-tools not installed. Run: pip install grpcio-tools", file=sys.stderr)
        sys.exit(1)


def clean_output():
    """Remove old generated files (keep __init__.py)."""
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.rglob("*_pb2*.py"):
            f.unlink()


def generate():
    """Run protoc to generate Python files."""
    from grpc_tools import protoc

    # protoc outputs mirror the import path structure, creating a proto/ subdir.
    # We generate into a temp location then move controls/ up one level.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        args = [
            "grpc_tools.protoc",
            f"--proto_path={IFACE_DIR}",
            f"--python_out={tmpdir}",
            f"--grpc_python_out={tmpdir}",
        ] + [str(IFACE_DIR / pf) for pf in PROTO_FILES]

        ret = protoc.main(args)
        if ret != 0:
            print(f"ERROR: protoc exited with code {ret}", file=sys.stderr)
            sys.exit(ret)

        # Move proto/controls/ -> OUTPUT_DIR/controls/
        src = Path(tmpdir) / "proto" / "controls"
        dst = OUTPUT_DIR / "controls"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def create_init_files():
    """Create __init__.py in every package directory."""
    for dirpath in sorted(OUTPUT_DIR.rglob("*")):
        if dirpath.is_dir():
            init = dirpath / "__init__.py"
            if not init.exists():
                init.touch()
    # Root __init__.py
    root_init = OUTPUT_DIR / "__init__.py"
    if not root_init.exists():
        root_init.touch()


def patch_module_names():
    """Patch generated code to use pacsys._proto.* module paths.

    Updates:
    - BuildTopDescriptorsAndMessages module name: proto.controls.X -> pacsys._proto.controls.X
    - Import statements in _grpc files: from proto.controls.X -> from pacsys._proto.controls.X

    protoc generates module names matching the .proto file import paths (proto/controls/...),
    but since we strip the proto/ prefix when copying to OUTPUT_DIR, Python module paths
    start at controls/. We patch proto.controls.* -> pacsys._proto.controls.* accordingly.
    """
    for pyfile in OUTPUT_DIR.rglob("*_pb2.py"):
        text = pyfile.read_text()
        # Patch BuildTopDescriptorsAndMessages module name
        patched = re.sub(
            r"(BuildTopDescriptorsAndMessages\(DESCRIPTOR, )'proto\.",
            r"\1'pacsys._proto.",
            text,
        )
        # Patch cross-file imports (e.g., DAQ_pb2 imports device_pb2)
        patched = patched.replace("from proto.", "from pacsys._proto.")
        if patched != text:
            pyfile.write_text(patched)

    for pyfile in OUTPUT_DIR.rglob("*_pb2_grpc.py"):
        text = pyfile.read_text()
        patched = text.replace("from proto.", "from pacsys._proto.")
        if patched != text:
            pyfile.write_text(patched)


def main():
    check_prerequisites()
    print(f"Generating proto files from {IFACE_DIR}")
    clean_output()
    generate()
    create_init_files()
    patch_module_names()

    generated = list(OUTPUT_DIR.rglob("*_pb2*.py"))
    print(f"Generated {len(generated)} files in {OUTPUT_DIR}")
    for f in sorted(generated):
        print(f"  {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
