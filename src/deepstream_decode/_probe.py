# SPDX-License-Identifier: Apache-2.0
"""Video metadata probe — used by consumers to pick equidistant frames
before submitting a decode request.

Uses ``pymediainfo`` (a Python binding around the upstream MediaInfo C++
library). Reads container metadata only — does not open a codec context,
does not decode frames. Cheap.

Public API:

    from deepstream_decode import probe_metadata
    meta = probe_metadata("/path/to/video.mp4")
    # meta is a VideoMetadata NamedTuple:
    #   meta.frame_count, meta.fps, meta.duration_sec, meta.width, meta.height
    # It also unpacks positionally for backward-compat with plain-tuple call sites:
    #   fc, fps, dur, w, h = probe_metadata(path)

Any unknown field is returned as 0 (matching the contract previously
established by vllm/multimodal/video.py::DeepStreamVideoBackend._probe_video_metadata).
"""

from __future__ import annotations

from typing import NamedTuple


class VideoMetadata(NamedTuple):
    """Container metadata for a video file."""
    frame_count: int
    fps: float
    duration_sec: float
    width: int
    height: int


def probe_metadata(filepath: str) -> VideoMetadata:
    """Read container metadata via ``pymediainfo``.

    Returns ``VideoMetadata(frame_count, fps, duration_sec, width, height)``,
    with any unknown value set to ``0``. Returns all-zeros if no Video track
    is present, the file doesn't exist, or MediaInfo can't parse it.
    """
    from pymediainfo import MediaInfo

    try:
        info = MediaInfo.parse(filepath)
    except Exception:
        return VideoMetadata(0, 0.0, 0.0, 0, 0)

    for track in info.tracks:
        if track.track_type != "Video":
            continue
        frame_count = int(track.frame_count or 0)
        fps_val = float(track.frame_rate or 0)
        duration_sec = float(track.duration or 0) / 1000.0
        width = int(track.width or 0)
        height = int(track.height or 0)
        # If the container didn't store an explicit frame count, derive it
        # from duration × fps. Same fallback as the previous vLLM-side
        # probe to preserve identical behavior across the swap.
        if frame_count == 0 and fps_val > 0 and duration_sec > 0:
            frame_count = round(fps_val * duration_sec)
        return VideoMetadata(frame_count, fps_val, duration_sec, width, height)

    return VideoMetadata(0, 0.0, 0.0, 0, 0)
