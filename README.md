# deepstream-decode

GPU-resident video decode via NVIDIA DeepStream — single-wheel install
bundles the DS shared libraries in a flat `_libs/` directory and
configures GStreamer + libv4l plugin discovery on Python import.

## Install (consumer side)

```bash
apt update
apt install gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} \
            gstreamer1.0-libav python3-gi python3-gst-1.0 libv4l-0 \
            cuda-libraries-13-0
pip install deepstream_decode-*.whl
```

That's it. Python consumers (vLLM, custom training stacks) get a working
decode pipeline on `import deepstream_decode`: `GST_PLUGIN_PATH` and
`LIBV4L2_PLUGIN_DIR` are set in-process to the bundled `_libs/`
directory.

## Quickstart

Once installed, confirm the pipeline works:

```bash
# Built-in selftest — verifies lib resolution, plugin discovery, CUDA context.
deepstream-decode-selftest
```

Sample decode app (lives in `examples/` in this repo, not shipped in the
wheel):

```bash
# Default: decodes the bundled sample 10s 1080p H.264 clip on the host
python3 examples/decode_example.py

# Decode any file you have
python3 examples/decode_example.py /path/to/video.mp4

# File + live RTSP source in one run
python3 examples/decode_example.py /path/to/video.mp4 \
    --rtsp rtsp://10.24.217.130:8554/

# Tune worker count and sampled-frames-per-call
python3 examples/decode_example.py video.mp4 --workers 4 --frames 16
```

Successful output ends with `frames shape : (N, H, W, 3)` on
`torch.uint8, cuda:0` — the GPU tensor is ready for downstream
consumers (model inference, training pipeline, etc.) with no D2H copy.

## Public API

```python
from deepstream_decode import (
    DecodePool,        # pool of N file-decode pipelines on N threads
    StreamHandle,      # one persistent pipeline for an RTSP/URI stream
    DecodeFrames,      # @dataclass: frames, n_kept, n_total, fps, error
    probe_metadata,    # cheap container-metadata probe (no decode)
    VideoMetadata,     # NamedTuple: frame_count, fps, duration_sec, width, height
    lib_dir,           # path to the bundled _libs/ directory
)
```

### `probe_metadata(filepath) -> VideoMetadata`

Read a video file's container metadata (frame count, fps, duration,
resolution) **without** decoding any frames. Used by consumers to
compute equidistant frame indices before submitting a `DecodePool.decode()`
request.

```python
from deepstream_decode import probe_metadata

meta = probe_metadata("/path/to/video.mp4")
print(meta.frame_count, meta.fps, meta.duration_sec, meta.width, meta.height)

# Backward-compatible with plain-tuple unpacking:
frame_count, fps, dur, w, h = probe_metadata("/path/to/video.mp4")

# Compute equidistant frame indices for an 8-frame sample:
import numpy as np
indices = np.linspace(0, meta.frame_count - 1, 8, dtype=int).tolist()
```

Backed by `libmediainfo` (auto-installed as a wheel dependency). Reads
the container index only — no codec context, no frame decode. Typical
cost: <10 ms per file.

Any unknown field is returned as `0`.

## What ships in the wheel

```
deepstream_decode/
├── __init__.py
├── _ds_dec.py        # DecodePool / StreamHandle API
├── _runtime.py       # _libs/ path resolver
├── _selftest.py      # `deepstream-decode-selftest` CLI
├── _version.py
└── _libs/                          # main libs + GStreamer plugins (flat)
    ├── libnvbufsurface.so
    ├── libnvbufsurftransform.so    (~26 MB)
    ├── libnvbuf_fdmap.so
    ├── libnvds_meta.so
    ├── libnvdsbufferpool.so
    ├── libnvdsgst_helper.so
    ├── libnvdsgst_meta.so
    ├── libgstnvdsseimeta.so
    ├── libgstnvcustomhelper.so
    ├── libnvv4l2.so
    ├── libcuvidv4l2.so
    ├── libv4l2.so.0                (symlink → libnvv4l2.so)
    ├── libgstnvvideo4linux2.so     (GStreamer plugin)
    ├── libgstnvvideoconvert.so     (GStreamer plugin)
    └── v4l_plugins/                # isolated to silence libv4l2's plugin scanner
        └── libcuvidv4l2_plugin.so
```

RPATHs: `$ORIGIN` on the flat libs (siblings); `$ORIGIN/..` on
`v4l_plugins/*.so` (their deps live one level up). The libv4l plugin
gets its own sub-dir because `libnvv4l2.so`'s scanner `dlsym`s every
`.so` in `LIBV4L2_PLUGIN_DIR` looking for the `libv4l2_plugin` symbol —
putting non-plugin libs alongside it triggers noisy `dlsym failed`
warnings.

## Building the wheel

1. Get the DeepStream tbz2 from NVIDIA. Two paths:
   - **Auto-download from NGC** (recommended): set `NGC_API_KEY` (https://ngc.nvidia.com
     → Setup → API Key) and the script fetches it on first run. URL pattern:
     `https://api.ngc.nvidia.com/v2/recipes/nvidia/deepstream/versions/<MAJOR.MINOR>/files/deepstream_sdk_v<VERSION>_x86_64.tbz2`
   - **Manual download**: developer.nvidia.com → DeepStream → Linux x86_64
     → Tar Package. Save next to `build_wheel.sh` (or pass
     `DS_TBZ2=/path/to/file.tbz2`). License-gated — not redistributable.
2. The patched `libnvv4l2.so` is committed to this folder (it has the
   `LIBV4L2_PLUGIN_DIR` env-var support that the SDK's stock libnvv4l2.so
   lacks). Override with `PATCHED_LIBNVV4L2=/path/to/it` if needed. This
   is a temporary substitution until the patched build ships in the DS
   tbz2 itself — at which point delete the file and set
   `USE_PATCHED_LIBNVV4L2=0`-equivalent in the script (or just remove
   the override block).
3. Run:

```bash
./build_wheel.sh                  # uses DS_VERSION=9.0.0 by default
./build_wheel.sh --ds-version=9.1.0
DS_TBZ2=/path/file.tbz2 ./build_wheel.sh
./build_wheel.sh --ds-src=/path/to/already_extracted_tree
```

Output: `dist/deepstream_decode-<VERSION>+cuda<MAJOR>-py3-none-manylinux_2_28_x86_64.whl`

Build prerequisites:

```bash
sudo apt update
sudo apt install patchelf binutils tar python3 python3-pip
pip install uv build
```

## Troubleshooting

### `GStreamer element creation failed: ['nvdec', 'nvvconv']`

GStreamer caches a plugin registry at `~/.cache/gstreamer-1.0/registry.<arch>.bin`.
If the cache was built **before** `cuda-libraries-13-0` was installed,
it permanently records "this plugin failed to load" — GStreamer trusts
the cache and never retries.

Common trigger: installing the wheel and running it once *before*
running the apt-install line, then installing `cuda-libraries-13-0`
afterwards. The first run cached the failure; subsequent runs don't
rescan even though the missing lib is now present.

Fix:

```bash
rm -rf ~/.cache/gstreamer-1.0/
python3 -c "import deepstream_decode"   # forces a rescan; rebuilds the cache
gst-inspect-1.0 nvv4l2decoder           # should now print Factory Details
```

### `libnppig.so.13: cannot open shared object file: No such file or directory`

CUDA's NPP image-processing runtime is missing. The wheel bundles
DeepStream's own libs but **not** the CUDA runtime — that comes from
apt:

```bash
apt update && apt install -y --no-install-recommends cuda-libraries-13-0
```

Other CUDA-side libs in the same package (`libcublas`, `libcudart`,
`libcuda`) are required too — they're all in `cuda-libraries-13-0`.

After installing, clear the GStreamer cache (see above) before re-running.

### `deepstream-decode-selftest` says "DeepStream libs not found"

The wheel's `_libs/` directory is empty or missing. Causes:

- The wheel was installed without `--force-reinstall` after a build
  that wiped the staging dir mid-build.
- Some packaging tools (older pip, certain CI runners) drop `.so`
  files from wheels by default.

Fix: `pip install --force-reinstall --no-deps deepstream_decode-*.whl`.
Verify with `python3 -c "import deepstream_decode, os; print(os.listdir(deepstream_decode.lib_dir()))"` — should list ~15 `.so` files.

### `Opening in BLOCKING MODE` printed during decode

Informational message from `nvv4l2decoder` itself — not an error.
NVDEC negotiated a blocking I/O mode for the V4L2 capture buffer
(default for file decode). Silence with:

```bash
GST_DEBUG=2 python3 your_script.py     # below WARNING level
```

### `dlsym failed: libcuvidv4l2.so: undefined symbol: libv4l2_plugin`

Older builds of this wheel placed `libcuvidv4l2_plugin.so` alongside
the main libs, triggering `libnvv4l2.so`'s plugin scanner to `dlsym`
every `.so` in `LIBV4L2_PLUGIN_DIR`. Current builds isolate it in
`_libs/v4l_plugins/`, so this warning shouldn't appear. If you see it,
you're on an old wheel — rebuild and reinstall.
