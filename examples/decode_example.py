#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sample app — decode a video file (and optionally an RTSP stream) via
deepstream_decode, then print info about the resulting GPU tensors.

Run:
    python3 decode_example.py <video.mp4>
    python3 decode_example.py <video.mp4> --rtsp rtsp://<host>:<port>/<path>
    python3 decode_example.py                      # uses the bundled sample

Requirements:
    - GPU-enabled host with NVIDIA driver
    - apt: gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly}
           gstreamer1.0-libav python3-gi python3-gst-1.0 libv4l-0
           cuda-libraries-13-0
    - pip install <the deepstream-decode wheel>
    - pip install torch (matching your CUDA version)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# The import triggers _ds_dec._preload_deepstream_libs(), which:
#   - resolves the bundled `_libs/` dir under site-packages
#   - sets GST_PLUGIN_PATH and LIBV4L2_PLUGIN_DIR
#   - ctypes-preloads the foundation DS libs with RTLD_GLOBAL
# After this single import, GStreamer can build pipelines using the
# DeepStream plugins.
import deepstream_decode


def show_setup() -> None:
    """Print what the wheel configured at import time."""
    print("┌─ deepstream_decode setup ─────────────────────────────────────")
    print(f"│ version            : {deepstream_decode.__version__}")
    print(f"│ lib_dir            : {deepstream_decode.lib_dir()}")
    print(f"│ GST_PLUGIN_PATH    : {os.environ.get('GST_PLUGIN_PATH', '<unset>')}")
    print(f"│ LIBV4L2_PLUGIN_DIR : {os.environ.get('LIBV4L2_PLUGIN_DIR', '<unset>')}")
    print("└───────────────────────────────────────────────────────────────")


def decode_file(path: str, num_workers: int = 2, max_frames: int = 8) -> None:
    """File-decode path — DecodePool builds N appsrc→nvv4l2decoder pipelines
    on N daemon threads. The file is read into memory and pushed as bytes;
    the requested frame indices are kept (exactly N). Each .decode() call
    picks a free worker."""
    import numpy as np

    from deepstream_decode import DecodePool, probe_metadata

    print(f"\n=== File decode: {path} ===")
    print(f"  workers={num_workers}, max_frames={max_frames}")

    with open(path, "rb") as fh:
        data = fh.read()

    # Probe metadata (GStreamer, no decode) for the frame count, then take a
    # uniform sample — the same shape a real consumer (e.g. vLLM) produces.
    fc, fps, dur, w, h, codec = probe_metadata(data)
    indices = np.linspace(0, max(fc - 1, 0), max_frames, dtype=int).tolist()

    pool = DecodePool(num_workers=num_workers)
    try:
        t0 = time.time()
        result = pool.decode(
            data, target_indices=indices, codec=codec, max_frames=max_frames,
            timeout_sec=30.0)
        dt = time.time() - t0

        if result.error:
            print(f"  ERROR: {result.error}")
            return

        f = result.frames
        print(f"  status        : OK")
        print(f"  elapsed       : {dt * 1000:.1f} ms")
        print(f"  frames shape  : {tuple(f.shape) if f is not None else None}")
        print(f"  dtype/device  : {f.dtype}, {f.device}" if f is not None else "")
        print(f"  n_kept        : {result.n_kept}")
        print(f"  n_total       : {result.n_total}")
        print(f"  fps           : {result.fps:.2f}")

        # Sanity-check: frames should be NHWC uint8 on cuda:0.
        if f is not None:
            assert f.dim() == 4, f"expected (N, H, W, C), got {f.shape}"
            assert f.shape[-1] == 3, f"expected RGB (3 channels), got C={f.shape[-1]}"
            assert f.dtype.is_floating_point is False, "expected uint8"
            assert "cuda" in str(f.device), "expected CUDA tensor"
            print(f"  ✓ frame layout valid (NHWC uint8 cuda)")

    finally:
        pool.shutdown()


def decode_rtsp(uri: str, num_segments: int = 3, max_frames: int = 8) -> None:
    """RTSP-stream path — StreamHandle keeps one persistent pipeline alive
    and yields a segment of frames per .decode_segment() call."""
    from deepstream_decode import StreamHandle

    print(f"\n=== RTSP stream: {uri} ===")
    print(f"  segments={num_segments}, max_frames={max_frames}")

    handle = StreamHandle(uri)
    try:
        for i in range(num_segments):
            t0 = time.time()
            result = handle.decode_segment(max_frames=max_frames, timeout_sec=30.0)
            dt = time.time() - t0

            if result.error:
                print(f"  [seg {i}] ERROR: {result.error}")
                break

            f = result.frames
            shape = tuple(f.shape) if f is not None else None
            print(f"  [seg {i}] {dt * 1000:6.1f} ms  shape={shape}  fps={result.fps:.2f}")

    finally:
        handle.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "video",
        nargs="?",
        default="/work/forked_vllm/sample_1080p_h264_10s_gop30.mp4",
        help="Path or URI to a video file (default: bundled 10s 1080p H.264 sample)",
    )
    p.add_argument(
        "--rtsp",
        default=None,
        help="Optional RTSP URI to also exercise StreamHandle "
             "(e.g., rtsp://10.24.217.130:8554/)",
    )
    p.add_argument("--workers", type=int, default=2, help="Decode pool workers (default 2)")
    p.add_argument("--frames", type=int, default=8, help="Max frames per decode call (default 8)")
    args = p.parse_args()

    show_setup()

    if not os.path.exists(args.video) and "://" not in args.video:
        print(f"\nERROR: video not found: {args.video}", file=sys.stderr)
        print("Pass a path or URI as the first argument.", file=sys.stderr)
        return 1

    decode_file(args.video, num_workers=args.workers, max_frames=args.frames)

    if args.rtsp:
        decode_rtsp(args.rtsp, max_frames=args.frames)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
