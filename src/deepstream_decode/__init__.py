# SPDX-License-Identifier: Apache-2.0
"""GPU-resident video decode via NVIDIA DeepStream.

Public API:

    from deepstream_decode import DecodePool, StreamHandle, DecodeFrames

The wheel bundles every DeepStream `.so` it needs in a single flat
directory at ``<site-packages>/deepstream_decode/_libs/``. On import, two
environment variables are configured for in-process consumers:

  * ``GST_PLUGIN_PATH``     — prepended with ``_libs/`` so GStreamer's
    scanner finds ``libgstnvvideo4linux2.so`` and
    ``libgstnvvideoconvert.so``.
  * ``LIBV4L2_PLUGIN_DIR``  — set to ``_libs/`` so the patched
    libnvv4l2.so finds ``libcuvidv4l2_plugin.so``.

Both point at the same path because the wheel keeps everything flat:

    libgstnvvideo4linux2.so      ← GStreamer plugin
        dlopens libv4l2.so.0
                ↓
    libv4l2.so.0                 ← symlink to libnvv4l2.so
        scans LIBV4L2_PLUGIN_DIR
                ↓
    libcuvidv4l2_plugin.so       ← libv4l plugin
        dlopens libcuvidv4l2.so
                ↓
    libcuvidv4l2.so              ← NVDEC backend

All those files sit in the same ``_libs/`` directory, sibling to each
other. RPATH=$ORIGIN means inter-lib NEEDED entries resolve to siblings.

Setup is done in-process — for Python consumers (vLLM, custom training
stacks), a single ``pip install`` + ``import deepstream_decode`` is
enough.  Standalone CLI tools like ``gst-inspect-1.0`` invoked from a
shell don't inherit Python's env, and would need an explicit export.
"""

# _ds_dec runs _preload_deepstream_libs() at module import time, which
# sets GST_PLUGIN_PATH + LIBV4L2_PLUGIN_DIR via _runtime helpers and
# ctypes-preloads the foundation libs.
from ._ds_dec import DecodeFrames, DecodePool, StreamHandle
from ._probe import VideoMetadata, probe_metadata
from ._runtime import lib_dir, plugin_dir
from ._version import __version__

__all__ = [
    "DecodeFrames",
    "DecodePool",
    "StreamHandle",
    "VideoMetadata",
    "__version__",
    "lib_dir",
    "plugin_dir",
    "probe_metadata",
]
