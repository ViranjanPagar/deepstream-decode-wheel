# SPDX-License-Identifier: Apache-2.0
"""Resolve the bundled `_libs/` directory.

Everything ships flat in one location — main libs, GStreamer plugins,
libv4l plugins. So `lib_dir()` and `plugin_dir()` return the same path.
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_BUNDLED_LIB_DIR = _PKG_DIR / "_libs"

_SENTINEL_LIB = "libnvbufsurface.so"


def _find_lib_dir() -> str:
    override = os.environ.get("DEEPSTREAM_DECODE_LIB_DIR")
    if override:
        if not (Path(override) / _SENTINEL_LIB).exists():
            raise RuntimeError(
                f"DEEPSTREAM_DECODE_LIB_DIR={override!r} does not contain "
                f"{_SENTINEL_LIB}."
            )
        return override

    if (_BUNDLED_LIB_DIR / _SENTINEL_LIB).exists():
        return str(_BUNDLED_LIB_DIR)

    raise RuntimeError(
        f"DeepStream libs not found at {_BUNDLED_LIB_DIR}. Reinstall with "
        "`pip install --force-reinstall deepstream-decode`, or set "
        f"DEEPSTREAM_DECODE_LIB_DIR to a directory containing {_SENTINEL_LIB}."
    )


def lib_dir() -> str:
    """Absolute path to the bundled `_libs/` directory.

    All DS shared libraries, GStreamer plugins, and the libv4l plugin
    live here in a flat layout."""
    return _find_lib_dir()


# Plugins live in the same directory as the main libs — the wheel keeps
# everything flat. Both GStreamer and libnvv4l2 scan this single path.
plugin_dir = lib_dir


def gst_plugin_dir() -> str:
    """Where GStreamer should look for our plugins. Same path as `lib_dir()`
    — the wheel keeps everything flat."""
    return _find_lib_dir()


def v4l_plugin_dir() -> str:
    """Where libnvv4l2.so should look for libcuvidv4l2_plugin.so.

    Lives in a dedicated `v4l_plugins/` sub-dir of the main lib dir.
    Reason: libnvv4l2.so's plugin scanner dlopens + dlsyms EVERY .so in
    LIBV4L2_PLUGIN_DIR looking for the `libv4l2_plugin` symbol.
    Pointing it at the flat `_libs/` triggers noisy `dlsym failed`
    warnings for every non-plugin .so the scanner tries. Isolating
    real libv4l plugins in a sub-dir keeps the scan quiet."""
    return os.path.join(_find_lib_dir(), "v4l_plugins")
