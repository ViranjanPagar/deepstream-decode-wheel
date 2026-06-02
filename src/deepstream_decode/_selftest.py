# SPDX-License-Identifier: Apache-2.0
"""Verify a fresh `deepstream-decode` install is functional.

Run after `apt install ./deepstream-decode_*.deb`. Walks five checks:

    1. Bundled DeepStream .so files present at the expected lib dir
    2. Bundled GStreamer plugins present
    3. System symlinks created by the .deb postinst
    4. ctypes can dlopen the .so files with RTLD_GLOBAL
    5. GStreamer + DeepStream elements (nvv4l2decoder, nvvideoconvert) are
       registered and discoverable

Exit code 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path

from . import _runtime

REQUIRED_LIBS = [
    "libnvbufsurface.so",
    "libnvbufsurftransform.so",
    "libnvbuf_fdmap.so",
    "libnvds_meta.so",
    "libnvdsgst_meta.so",
    "libnvdsbufferpool.so",
    "libnvdsgst_helper.so",
    "libgstnvdsseimeta.so",
    "libgstnvcustomhelper.so",
    "libnvv4l2.so",
    "libcuvidv4l2.so",
]
REQUIRED_GST_PLUGINS = ["libgstnvvideo4linux2.so", "libgstnvvideoconvert.so"]
REQUIRED_SYMLINKS = [
    "/usr/lib/x86_64-linux-gnu/libv4l2.so.0",
    "/usr/lib/x86_64-linux-gnu/libv4l/plugins/libcuvidv4l2_plugin.so",
]

# NPP runtime libs the DS plugins call into. Probed at runtime to verify
# the host has a matching-CUDA-major NPP package installed.
NPP_REQUIRED_SOS = ("libnppig", "libnppidei", "libnppc")


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _detect_bundled_cuda_major(lib_dir: Path) -> int | None:
    """Read NPP SONAME from the bundled libnvbufsurftransform.so to find
    out which CUDA major the bundled .so files were built against.

    Tries `objdump` first (binutils), then `readelf` (also binutils).
    Returns None if neither is available — caller falls back gracefully.
    """
    target = lib_dir / "libnvbufsurftransform.so"
    if not target.exists():
        return None
    for cmd in (("objdump", "-p"), ("readelf", "-d")):
        try:
            out = subprocess.check_output(
                [*cmd, str(target)],
                text=True, stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue
        for line in out.splitlines():
            m = re.search(r"libnppig\.so\.(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def main() -> int:
    fails = 0

    try:
        lib_dir = Path(_runtime.lib_dir())
    except RuntimeError as e:
        _fail(str(e))
        return 1

    print("[1/6] Bundled .so files")
    for so in REQUIRED_LIBS:
        p = lib_dir / so
        if p.exists():
            _ok(f"{p}")
        else:
            _fail(f"{p}  (MISSING)")
            fails += 1

    print("[2/6] Bundled GStreamer plugins")
    for so in REQUIRED_GST_PLUGINS:
        p = lib_dir / "gst-plugins" / so
        if p.exists():
            _ok(f"{p}")
        else:
            _fail(f"{p}  (MISSING)")
            fails += 1

    print("[3/6] System symlinks (created by `apt install` postinst)")
    for path in REQUIRED_SYMLINKS:
        p = Path(path)
        if p.is_symlink() or p.exists():
            _ok(f"{p}")
        else:
            _fail(f"{p}  (re-run: sudo apt install --reinstall deepstream-decode)")
            fails += 1

    print("[4/6] ctypes dlopen of bundled libs")
    for so in REQUIRED_LIBS:
        try:
            ctypes.CDLL(str(lib_dir / so), mode=ctypes.RTLD_GLOBAL)
            _ok(f"loaded {so}")
        except OSError as e:
            _fail(f"{so}: {e}")
            fails += 1

    print("[5/6] CUDA NPP libraries for bundled .so files")
    cuda_major = _detect_bundled_cuda_major(lib_dir)
    if cuda_major is None:
        _fail("could not read NPP SONAME from bundled .so "
              "(install `binutils` for objdump/readelf, or check the bundle)")
        fails += 1
    else:
        _ok(f"bundled .so files target CUDA {cuda_major}.x")
        for stem in NPP_REQUIRED_SOS:
            so_name = f"{stem}.so.{cuda_major}"
            try:
                ctypes.CDLL(so_name)
                _ok(f"{so_name} loadable")
            except OSError:
                _fail(
                    f"{so_name} not found — install `libnpp-{cuda_major}-X` "
                    f"(e.g. sudo apt install libnpp-{cuda_major}-0)"
                )
                fails += 1

    print("[6/6] GStreamer + DeepStream elements registered")
    try:
        import gi  # type: ignore[import-not-found]

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore[import-not-found]

        os.environ["GST_PLUGIN_PATH"] = (
            _runtime.gst_plugin_dir() + os.pathsep
            + os.environ.get("GST_PLUGIN_PATH", "")
        )
        Gst.init(None)
        for elem in ("appsrc", "parsebin", "nvv4l2decoder", "nvvideoconvert"):
            if Gst.ElementFactory.find(elem):
                _ok(f"element '{elem}' available")
            else:
                _fail(f"element '{elem}' NOT FOUND")
                fails += 1
    except Exception as e:
        _fail(f"GStreamer init failed: {e}")
        fails += 1

    print()
    if fails == 0:
        print("\033[32mAll checks passed\033[0m")
        return 0
    print(f"\033[31m{fails} check(s) failed\033[0m")
    return 1


if __name__ == "__main__":
    sys.exit(main())
