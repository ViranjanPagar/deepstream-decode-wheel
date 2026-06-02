# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepStream decode pool — GStreamer pipelines + CUDA frame capture.

N pipelines run on N daemon threads inside a single process, all
sharing one CUDA context. Each pipeline's buffer probes copy decoded
NVMM frames into a CUDA tensor allocated from PyTorch's caching
allocator and the caller uses that tensor directly — no D2H/H2D
round-trip, no IPC handle, no per-decode tensor reconstruction.

Two pool shapes are exposed:

* :class:`DecodePool` — file-decode pool. Workers build an
  ``appsrc → parsebin → nvv4l2decoder → nvvideoconvert →
  capsfilter[NVMM RGB] → fakesink`` pipeline and push raw container
  bytes per request (no file path). Frames are selected by index (exact,
  1:1, full decode) or by PTS with parser-level GOP-drop, chosen per
  ``decode()`` call. ``parsebin`` auto-routes H.264, H.265, and the
  containers wrapping them (MP4, MKV, MPEG-TS).
* :class:`StreamHandle` — one persistent pipeline per RTSP/URI stream
  (``uridecodebin → nvvideoconvert → capsfilter → fakesink``).
"""

from __future__ import annotations

import bisect
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DeepStream runtime preload (matches deepstream_runtime/decode_demo.py)
# ----------------------------------------------------------------------
# Loaded at module import — both the parent and any spawned workers re-run
# this when they import the module, so GStreamer's nv* plugins can resolve
# their nvbufsurface / nvds* deps without LD_LIBRARY_PATH manipulation.
def _preload_deepstream_libs() -> None:
    try:
        from . import _runtime
    except ImportError:
        return
    try:
        ds_lib_dir = _runtime.lib_dir()
    except RuntimeError as e:
        logger.warning("[ds_decode_pool] DeepStream libs not located: %s", e)
        return
    lib_dir = Path(ds_lib_dir)
    for soname in (
        "libnvbuf_fdmap.so",
        "libnvbufsurface.so",
        "libnvbufsurftransform.so",
        "libnvdsbufferpool.so",
        "libnvdsgst_helper.so",
        "libnvdsgst_meta.so",
        "libnvds_meta.so",
        "libgstnvdsseimeta.so",
        "libgstnvcustomhelper.so",
        "libcuvidv4l2.so",
        "libnvv4l2.so",
    ):
        path = lib_dir / soname
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                logger.warning("[ds_decode_pool] preload failed: %s (%s)",
                               path, e)
    os.environ["GST_PLUGIN_PATH"] = (
        _runtime.gst_plugin_dir() + os.pathsep
        + os.environ.get("GST_PLUGIN_PATH", "")
    )
    # libnvv4l2.so reads LIBV4L2_PLUGIN_DIR to locate libcuvidv4l2_plugin.so.
    # The wheel keeps both files in the same flat _libs/ directory.
    os.environ["LIBV4L2_PLUGIN_DIR"] = _runtime.v4l_plugin_dir()


_preload_deepstream_libs()


# ----------------------------------------------------------------------
# NvBufSurface ctypes layout
# ----------------------------------------------------------------------
# Mirrors the public ``nvbufsurface.h`` definitions for DeepStream 9.x.
# We only need to read pitch + dataPtr from surfaceList[0]; the trailing
# fields are present so sizeof() lines up with the C struct, but their
# values are ignored.
_NVBUF_MAX_PLANES = 4
_STRUCTURE_PADDING = 4


class _NvBufSurfacePlaneParams(ctypes.Structure):
    _fields_ = [
        ("num_planes",  ctypes.c_uint32),
        ("width",       ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("height",      ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("pitch",       ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("offset",      ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("psize",       ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("bytesPerPix", ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("_reserved",   ctypes.c_void_p * (_STRUCTURE_PADDING
                                           * _NVBUF_MAX_PLANES)),
    ]


class _NvBufSurfaceMappedAddr(ctypes.Structure):
    _fields_ = [
        ("addr",      ctypes.c_void_p * _NVBUF_MAX_PLANES),
        ("eglImage",  ctypes.c_void_p),
        ("nvmmPtr",   ctypes.c_void_p),
        ("cudaPtr",   ctypes.c_void_p),
        ("_reserved", ctypes.c_void_p * _STRUCTURE_PADDING),
    ]


class _NvBufSurfaceParams(ctypes.Structure):
    _fields_ = [
        ("width",       ctypes.c_uint32),
        ("height",      ctypes.c_uint32),
        ("pitch",       ctypes.c_uint32),
        ("colorFormat", ctypes.c_int),
        ("layout",      ctypes.c_int),
        ("bufferDesc",  ctypes.c_uint64),
        ("dataSize",    ctypes.c_uint32),
        ("dataPtr",     ctypes.c_void_p),
        ("planeParams", _NvBufSurfacePlaneParams),
        ("mappedAddr",  _NvBufSurfaceMappedAddr),
        ("paramex",     ctypes.c_void_p),
        ("cudaBuffer",  ctypes.c_void_p),
        ("_reserved",   ctypes.c_void_p * _STRUCTURE_PADDING),
    ]


class _NvBufSurface(ctypes.Structure):
    _fields_ = [
        ("gpuId",         ctypes.c_uint32),
        ("batchSize",     ctypes.c_uint32),
        ("numFilled",     ctypes.c_uint32),
        ("isContiguous",  ctypes.c_bool),
        ("memType",       ctypes.c_int),
        ("surfaceList",   ctypes.POINTER(_NvBufSurfaceParams)),
        ("isImportedBuf", ctypes.c_bool),
        ("_reserved",     ctypes.c_void_p * _STRUCTURE_PADDING),
    ]


# ----------------------------------------------------------------------
# Request / Result dataclasses (picklable)
# ----------------------------------------------------------------------
@dataclass
class _DecodeRequest:
    job_id: int
    uri: str = ""                   # RTSP/stream worker only
    data: bytes = b""               # file mode: raw container bytes (appsrc)
    codec: str = ""                 # codec hint (e.g. "h264"/"hevc") for reuse
    target_pts_ns: tuple = ()       # sorted PTS targets (PTS mode / GOP-drop)
    target_indices: tuple = ()      # sorted frame indices (index mode)
    use_pts_mode: bool = True       # True = PTS select, False = index select
    max_frames: int = 8
    timeout_sec: float = 30.0
    # Index-mode GOP-drop: drop whole GOPs containing no target frame at the
    # parser (before NVDEC) so the decoder only decodes the GOPs that matter.
    # Selection stays exact (by frame index). Needs ``fps`` to reconstruct
    # original display indices from PTS once frames are dropped. Most useful
    # for sparse sampling of long videos; ~no gain when N >= #GOPs.
    gop_drop: bool = False
    fps: float = 0.0                # decoded fps hint (for index<->PTS map)


@dataclass
class _DecodeResult:
    job_id: int
    worker_id: int
    n_kept: int = 0
    n_total: int = 0
    fps: float = 0.0
    error: str = ""
    # CUDA tensor (N, H, W, 3) uint8 written by the worker's CUDA stream.
    # Caller uses it directly — same address space.
    frames: Any = None


# ----------------------------------------------------------------------
# Common worker state — pipeline + probe context + CUDA stream
# ----------------------------------------------------------------------
class _BaseWorkerState:
    """Shared probe / capture logic for both file and RTSP workers."""

    def __init__(self, worker_id: int, drop_interval: int):
        self.worker_id = worker_id
        self.drop_interval = max(0, drop_interval)
        self.pipeline = None
        self.elements: dict[str, Any] = {}
        self._Gst = None
        self._gst_imported = False

        # CUDA / cudart bindings — created lazily after torch.cuda.init().
        self._cudart = None
        self._cuda_stream = ctypes.c_void_p(0)

        # Per-decode mutable probe state — reset by _reset_for_decode.
        self.target_pts: tuple[int, ...] = ()
        self.target_indices: tuple[int, ...] = ()
        self.use_pts_mode = True
        self.max_frames = 0
        self.kept = 0
        self.total = 0
        self.target_cursor = 0
        self.early_eos = False
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.has_error = False
        self.err_msg = ""

        # Pre-allocated destination tensor for this decode (N, H, W, 3).
        # Allocated on the first kept frame — H/W are unknown until then.
        self.dst_tensor: torch.Tensor | None = None
        self.dst_ptr = 0
        self.frame_bytes = 0
        self.dst_pitch = 0

        # GOP-aware parser drop state (file mode only).
        self.gop_initialized = False
        self.gop_i_pts = 0
        self.gop_duration = 0
        # Cached "any unmatched target falls in this GOP" decision, set
        # on each I-frame and re-evaluated on deltas as the select probe
        # consumes targets. Defaults to True until the first I-frame so
        # we don't drop an opening fragment of P-frames before any I-frame
        # has been parsed.
        self.gop_has_target = True
        # Inferred frame interval (next-delta-pts − last-i-frame-pts).
        # Reset on every I-frame, set on the first delta after it.
        self.frame_duration = 0

        # Index-mode GOP-drop state. ``frame_count`` is the running
        # decode-order compressed-frame count at the parser (== display index
        # at GOP boundaries, since GOPs are contiguous). ``gop_size`` is the
        # largest GOP length seen so far (frames). ``base_pts`` is the PTS of
        # original display index 0, derived from a keyframe so the output side
        # can map PTS -> original index even after frames are dropped
        # (offset-invariant). ``frame_dur_ns`` = round(1e9 / fps).
        self.gop_drop = False
        self.frame_count = 0
        self.gop_size = 0
        self.gop_start_idx = 0
        self.base_pts = None
        self.frame_dur_ns = 0

        # True once the pipeline has actually streamed (PLAYING run to
        # EOS or a prior decode loop). Used by the file worker to skip
        # the no-op FLUSH-seek to byte 0 on a freshly-built pipeline,
        # avoiding the seek/PAUSED-transition race.
        self.pipeline_has_streamed = False

        # RTSP-only "decode complete" signalling. Live pipelines never
        # accept EOS; the count probe sets this once max_frames is hit.
        self._done_event = None

    # ------------------------------------------------------------------
    # Lazy GStreamer / cudart setup
    # ------------------------------------------------------------------
    def _ensure_gst(self):
        if self._gst_imported:
            return self._Gst
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore
        Gst.init(None)
        self._Gst = Gst
        self._gst_imported = True
        return Gst

    def _ensure_cudart(self):
        if self._cudart is not None:
            return self._cudart
        lib = ctypes.CDLL("libcudart.so")
        # cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
        #     const void* src, size_t spitch, size_t width, size_t height,
        #     enum cudaMemcpyKind kind, cudaStream_t stream);
        lib.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int,    ctypes.c_void_p,
        ]
        lib.cudaMemcpy2DAsync.restype = ctypes.c_int
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaStreamCreate.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        if lib.cudaStreamCreate(ctypes.byref(self._cuda_stream)) != 0:
            raise RuntimeError("cudaStreamCreate failed")
        self._cudart = lib
        return lib

    # ------------------------------------------------------------------
    # Per-decode reset
    # ------------------------------------------------------------------
    def _reset_for_decode(self, req: _DecodeRequest) -> None:
        self.target_pts = tuple(sorted(req.target_pts_ns))
        self.target_indices = tuple(sorted(req.target_indices))
        self.use_pts_mode = req.use_pts_mode
        self.max_frames = max(1, req.max_frames)
        self.kept = 0
        self.total = 0
        self.target_cursor = 0
        self.early_eos = False
        self.fps = 0.0
        # Width/height are sticky across decodes for the same pipeline —
        # only clear them when the source URI changes (caller's job).
        self.has_error = False
        self.err_msg = ""
        self.dst_tensor = None
        self.dst_ptr = 0
        self.frame_bytes = 0
        self.dst_pitch = 0
        self.gop_initialized = False
        self.gop_i_pts = 0
        self.gop_duration = 0
        self.gop_has_target = True
        self.frame_duration = 0
        # Index-mode GOP-drop setup. Requires fps to map PTS<->index once
        # frames are dropped; disable (fall back to full decode) if absent.
        self.gop_drop = bool(req.gop_drop) and not req.use_pts_mode
        self.frame_count = 0
        self.gop_size = 0
        self.gop_start_idx = 0
        self.base_pts = None
        self.frame_dur_ns = int(round(1e9 / req.fps)) if req.fps > 0 else 0
        if self.gop_drop and self.frame_dur_ns <= 0:
            logger.warning(
                "[ds w%d] gop_drop requested without fps — decoding full "
                "stream instead", self.worker_id)
            self.gop_drop = False
        self.debug_pts = os.environ.get("DS_DEBUG_PTS", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if self._done_event is not None:
            self._done_event.clear()

    # ------------------------------------------------------------------
    # Probe callbacks — all accept the GIL implicitly
    # ------------------------------------------------------------------
    def _parser_probe(self, pad, info, _ud):
        """Parser src pad — drop GOPs containing no target PTS.

        On each I-frame we cache whether any unmatched target falls in
        the upcoming GOP and reuse that decision on the deltas — avoids
        a bisect per buffer. Deltas re-evaluate too: as the select
        probe consumes targets via ``target_cursor``, a GOP that
        started with a target may run out of them mid-GOP, in which
        case we drop the trailing deltas.
        """
        Gst = self._Gst
        if self.early_eos:
            return Gst.PadProbeReturn.DROP
        if self.gop_drop and self.target_indices:
            return self._parser_probe_index(info)
        if not self.use_pts_mode or not self.target_pts:
            return Gst.PadProbeReturn.OK

        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        pts = buf.pts
        is_keyframe = not (buf.get_flags() & Gst.BufferFlags.DELTA_UNIT)

        if is_keyframe:
            if self.gop_initialized and pts > self.gop_i_pts:
                gap = pts - self.gop_i_pts
                if gap > self.gop_duration:
                    self.gop_duration = gap
            self.gop_i_pts = pts
            self.gop_initialized = True
            # Frame interval is reinferred from the first delta of the
            # new GOP — codecs occasionally vary it across GOPs.
            self.frame_duration = 0
            self.gop_has_target = self._gop_contains_target(pts)
            return Gst.PadProbeReturn.OK

        # First delta after the I-frame: estimate frame interval. Used
        # below to ignore targets that land in the very last frame slot
        # of this GOP — they're matched by the next GOP's I-frame anyway.
        if self.frame_duration == 0 and self.gop_initialized:
            self.frame_duration = pts - self.gop_i_pts

        if not self.gop_initialized or self.gop_duration == 0:
            return Gst.PadProbeReturn.OK

        if not self.gop_has_target:
            return Gst.PadProbeReturn.DROP

        # Re-check from the current delta pts — once the select probe
        # has matched the last target in this GOP, drop the rest.
        self.gop_has_target = self._gop_contains_target(pts)
        if not self.gop_has_target:
            return Gst.PadProbeReturn.DROP

        return Gst.PadProbeReturn.OK

    def _gop_contains_target(self, lower_bound_pts: int) -> bool:
        """Whether an unmatched target falls within the current GOP.

        ``lower_bound_pts`` is the I-frame pts on a fresh GOP, or the
        current delta pts when re-checking mid-GOP. ``target_cursor``
        skips targets the select probe has already matched. The upper
        bound shrinks by ``frame_duration`` when known: a target sitting
        in the last frame slot is matched by the next GOP's I-frame and
        doesn't need this GOP's deltas at all.
        """
        gop_end = self.gop_i_pts + self.gop_duration
        upper = (gop_end - self.frame_duration
                 if self.frame_duration > 0 else gop_end)
        lo = max(self.target_cursor,
                 bisect.bisect_left(self.target_pts, lower_bound_pts))
        return lo < len(self.target_pts) and self.target_pts[lo] < upper

    # ------------------------------------------------------------------
    # Index-mode GOP-drop (parser side): drop whole GOPs with no target
    # frame so NVDEC never decodes them. Selection stays exact (by index).
    # ------------------------------------------------------------------
    def _gop_contains_target_idx(self, gop_start: int) -> bool:
        """Whether an unmatched target index falls in the current GOP.

        The GOP spans original display indices ``[gop_start, gop_start +
        gop_size)``, where ``gop_size`` is the largest GOP length seen so far.
        This is **exact for constant-GOP streams** (the common case — once the
        first inter-keyframe gap is learned, every boundary is known). The
        first GOP (size still unknown) is always kept. Caveat for *variable*
        GOPs: a GOP longer than any seen so far, with a target only in its
        tail beyond ``gop_size``, could be dropped and that frame lost — the
        same limitation the PTS GOP-drop path has. For exactness on highly
        variable GOPs a keyframe pre-scan would be needed.
        """
        upper = gop_start + (self.gop_size if self.gop_size > 0 else (1 << 62))
        lo = bisect.bisect_left(self.target_indices, gop_start)
        return lo < len(self.target_indices) and self.target_indices[lo] < upper

    def _parser_probe_index(self, info):
        """Parser src pad — index-mode GOP-drop.

        Counts compressed frames in decode order (``frame_count``); at each
        keyframe that count equals the GOP's first original *display* index
        (GOPs are contiguous in both orders). Keeps the GOP iff a target index
        falls in it, dropping every frame of an empty GOP before the decoder.
        Also seeds ``base_pts`` (PTS of display index 0) from the first
        keyframe so the select probe can recover original indices from PTS.
        """
        Gst = self._Gst
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        is_key = not (buf.get_flags() & Gst.BufferFlags.DELTA_UNIT)
        if is_key:
            if self.gop_initialized and self.frame_count > self.gop_start_idx:
                gap = self.frame_count - self.gop_start_idx
                if gap > self.gop_size:
                    self.gop_size = gap
            self.gop_start_idx = self.frame_count
            self.gop_initialized = True
            if (self.base_pts is None and self.frame_dur_ns > 0
                    and buf.pts != Gst.CLOCK_TIME_NONE):
                # Keyframe is displayed at index gop_start_idx, so the PTS of
                # display index 0 is kf_pts - gop_start * frame_dur (cancels
                # any container PTS baseline offset).
                self.base_pts = buf.pts - self.gop_start_idx * self.frame_dur_ns
            self.gop_has_target = self._gop_contains_target_idx(
                self.gop_start_idx)
        self.frame_count += 1
        if self.gop_has_target:
            return Gst.PadProbeReturn.OK
        return Gst.PadProbeReturn.DROP

    def _select_probe(self, pad, info, _ud):
        """nvvideoconvert sink pad — pick which decoded frames to keep."""
        Gst = self._Gst
        if self.early_eos:
            return Gst.PadProbeReturn.DROP

        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        # First frame: capture fps from caps so the caller can report it.
        if self.total == 0:
            caps = pad.get_current_caps()
            if caps and caps.get_size():
                s = caps.get_structure(0)
                ok, num, den = s.get_fraction("framerate")
                if ok and den > 0:
                    self.fps = num / den

        self.total += 1

        if self.use_pts_mode:
            if not self.target_pts:
                return Gst.PadProbeReturn.OK
            pts = buf.pts
            if pts == Gst.CLOCK_TIME_NONE:
                return Gst.PadProbeReturn.OK
            if self.target_cursor >= len(self.target_pts):
                return Gst.PadProbeReturn.DROP
            if pts < self.target_pts[self.target_cursor]:
                return Gst.PadProbeReturn.DROP
            while (self.target_cursor < len(self.target_pts)
                   and self.target_pts[self.target_cursor] <= pts):
                self.target_cursor += 1
        else:
            if self.target_indices:
                # With GOP-drop the decoder skips whole GOPs, so the output
                # frame count no longer equals the original display index —
                # recover it from PTS (offset-invariant via base_pts). Without
                # GOP-drop every frame is decoded in order, so the running
                # output count is the index directly.
                if (self.gop_drop and self.base_pts is not None
                        and self.frame_dur_ns > 0
                        and buf.pts != Gst.CLOCK_TIME_NONE):
                    idx = int(round(
                        (buf.pts - self.base_pts) / self.frame_dur_ns))
                else:
                    idx = self.total - 1
                lo, hi = 0, len(self.target_indices) - 1
                hit = False
                while lo <= hi:
                    m = (lo + hi) >> 1
                    v = self.target_indices[m]
                    if v == idx:
                        hit = True
                        break
                    if v < idx:
                        lo = m + 1
                    else:
                        hi = m - 1
                if not hit:
                    return Gst.PadProbeReturn.DROP

        if self.kept >= self.max_frames:
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def _copy_probe(self, pad, info, _ud):
        """capsfilter src pad — copy NVMM RGB buffer into dst tensor."""
        Gst = self._Gst
        if self.early_eos or self.has_error:
            return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        if self.kept >= self.max_frames or self.max_frames == 0:
            return Gst.PadProbeReturn.OK

        src_ptr, src_pitch, p_width, p_height = _read_nvbuf_surface_first(
            buf, Gst)
        if not src_ptr:
            return Gst.PadProbeReturn.OK

        # Lazy-allocate destination tensor on the first kept frame.
        if self.dst_tensor is None:
            self.width = p_width
            self.height = p_height
            self.dst_pitch = self.width * 3
            self.frame_bytes = self.height * self.dst_pitch
            self.dst_tensor = torch.empty(
                (self.max_frames, self.height, self.width, 3),
                dtype=torch.uint8, device="cuda")
            self.dst_ptr = self.dst_tensor.data_ptr()

        dst = self.dst_ptr + self.kept * self.frame_bytes
        if not src_pitch:
            src_pitch = self.dst_pitch
        rc = self._cudart.cudaMemcpy2DAsync(
            ctypes.c_void_p(dst), self.dst_pitch,
            ctypes.c_void_p(src_ptr), src_pitch,
            self.dst_pitch, self.height,
            3,  # cudaMemcpyDeviceToDevice
            self._cuda_stream,
        )
        if rc != 0:
            self.has_error = True
            self.err_msg = f"cudaMemcpy2DAsync rc={rc}"
            return Gst.PadProbeReturn.OK

        self.kept += 1
        if self.debug_pts:
            logger.info(
                "[ds w%d] OUT keep #%d  frame_pts_ns=%d  (decoded so far=%d)",
                self.worker_id, self.kept - 1, buf.pts, self.total)
        if self.kept >= self.max_frames and not self.early_eos:
            self.early_eos = True
            self._on_max_frames_reached(pad)
        return Gst.PadProbeReturn.OK

    def _on_max_frames_reached(self, pad) -> None:
        """File mode pushes EOS; RTSP signals the worker thread instead."""
        Gst = self._Gst
        if self._done_event is not None:
            self._done_event.set()
        else:
            pad.push_event(Gst.Event.new_eos())

    # ------------------------------------------------------------------
    # Debug: dump kept frames to disk for visual inspection.
    # ------------------------------------------------------------------
    def _maybe_dump_frames(self, frames, req: "_DecodeRequest") -> None:
        """When ``DS_DUMP_DIR`` is set, write each kept frame as a PNG named
        ``w<worker>_job<id>_seq<n>_idx<original-index>.png`` so the selected
        frames (and their index mapping) can be eyeballed. Best-effort: any
        failure is logged and ignored so it never affects decode."""
        dump_dir = os.environ.get("DS_DUMP_DIR", "").strip()
        if not dump_dir or frames is None:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            arr = frames.detach().to("cpu").numpy()  # (N, H, W, 3) RGB uint8
            # Label each kept frame with the target index it satisfies, when
            # known (index mode); otherwise fall back to its sequence number.
            labels = (list(self.target_indices)
                      if self.target_indices else list(range(arr.shape[0])))
            try:
                from PIL import Image
                def _save(a, p):
                    Image.fromarray(a, "RGB").save(p)
            except Exception:
                import cv2
                def _save(a, p):
                    cv2.imwrite(p, a[:, :, ::-1])  # RGB -> BGR for cv2
            n = arr.shape[0]
            for j in range(n):
                idx = labels[j] if j < len(labels) else j
                fn = os.path.join(
                    dump_dir,
                    f"w{self.worker_id}_job{req.job_id}"
                    f"_seq{j:03d}_idx{idx:06d}.png")
                _save(arr[j], fn)
            logger.info("[ds w%d] dumped %d frames -> %s",
                        self.worker_id, n, dump_dir)
        except Exception as e:  # pragma: no cover - debug aid only
            logger.warning("[ds dump] failed to dump frames: %s", e)

    # ------------------------------------------------------------------
    # Build a populated _DecodeResult after the pipeline returns
    # ------------------------------------------------------------------
    def _finalize_result(self, req: _DecodeRequest,
                         error: str) -> _DecodeResult:
        if self._cudart is not None:
            self._cudart.cudaStreamSynchronize(self._cuda_stream)

        frames = None
        if self.dst_tensor is not None and self.kept > 0 and not self.has_error:
            # Slice down to the actual kept count, ``contiguous()`` so the
            # CUDA-IPC handoff carries only the populated rows.
            frames = self.dst_tensor[: self.kept].contiguous()
            self._maybe_dump_frames(frames, req)

        err = error or (self.err_msg if self.has_error else "")
        return _DecodeResult(
            job_id=req.job_id,
            worker_id=self.worker_id,
            n_kept=self.kept,
            n_total=self.total,
            fps=self.fps,
            error=err,
            frames=frames,
        )


# ----------------------------------------------------------------------
# File-mode worker — appsrc pipeline fed raw container bytes per request
# ----------------------------------------------------------------------
class _FileWorkerState(_BaseWorkerState):
    def __init__(self, worker_id: int, drop_interval: int):
        super().__init__(worker_id, drop_interval)
        self._probe_ids: dict[str, int] = {}
        # Codec the currently-warm nvv4l2decoder was built for. A new
        # stream with a different codec (e.g. H.265 after H.264) can't
        # reuse the session — see decode().
        self.current_codec: str | None = None

    # ------------------------------------------------------------------
    def ensure_pipeline(self) -> None:
        if self.pipeline is None:
            self._build()

    def _rebuild_all(self) -> None:
        """Full teardown + rebuild, including the nvv4l2decoder. Used when
        the incoming codec differs from the warm decoder's — the NVDEC
        session is codec-specific and cannot be reused across H.264/H.265.
        """
        Gst = self._Gst
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.get_state(5 * Gst.SECOND)
        self.pipeline = None
        self.elements = {}
        self._probe_ids = {}
        self._build()

    def _build(self) -> None:
        Gst = self._ensure_gst()
        self._ensure_cudart()

        # The downstream chain ``nvv4l2decoder -> nvvideoconvert ->
        # capsfilter[NVMM RGB] -> fakesink`` is built once and kept warm
        # across decodes. The NVDEC session + NVMM capture buffers held by
        # nvv4l2decoder are the expensive part, so they must NOT be torn
        # down between byte streams. Only the ``appsrc -> parsebin`` source
        # is recreated per stream (see _reset_source) — parsebin does not
        # cleanly accept a second stream, but the decoder downstream of it
        # survives a flush.
        # Frame selection is done by the Python ``_select_probe`` on the
        # nvvideoconvert sink pad — it keeps only the frames whose decode-order
        # index (or PTS, in GOP-drop mode) is in the target list and drops the
        # rest before color conversion, so nvvideoconvert only converts the
        # kept frames.
        # q_in decouples parsing (CPU, on the streaming thread) from NVDEC,
        # and q_dec decouples NVDEC from the select/convert/copy downstream —
        # so the decoder stays fed instead of stalling while the thread
        # parses the next frames or copies kept ones out.
        elems = {
            "q_in":     Gst.ElementFactory.make("queue",          None),
            "nvdec":    Gst.ElementFactory.make("nvv4l2decoder",  None),
            "q_dec":    Gst.ElementFactory.make("queue",          None),
            "nvvconv":  Gst.ElementFactory.make("nvvideoconvert", None),
            "capsf":    Gst.ElementFactory.make("capsfilter",     None),
            "sink":     Gst.ElementFactory.make("fakesink",       None),
        }
        missing = [k for k, v in elems.items() if v is None]
        if missing:
            raise RuntimeError(f"GStreamer element creation failed: {missing}")

        elems["nvdec"].set_property("drop-frame-interval", self.drop_interval)
        elems["nvdec"].set_property("num-extra-surfaces", 6)
        # Disable the queues' time/byte caps so they only bound on buffer
        # count (and, for q_dec, the NVMM surface pool) — no premature
        # blocking on the default 1s / 10MB limits.
        for q in ("q_in", "q_dec"):
            elems[q].set_property("max-size-time", 0)
            elems[q].set_property("max-size-bytes", 0)
        elems["nvvconv"].set_property("nvbuf-memory-type", 2)
        elems["sink"].set_property("sync", False)
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGB")
        elems["capsf"].set_property("caps", caps)

        pipeline = Gst.Pipeline.new(None)
        for e in elems.values():
            pipeline.add(e)

        if not (elems["q_in"].link(elems["nvdec"])
                and elems["nvdec"].link(elems["q_dec"])
                and elems["q_dec"].link(elems["nvvconv"])
                and elems["nvvconv"].link(elems["capsf"])
                and elems["capsf"].link(elems["sink"])):
            raise RuntimeError("downstream link failed")

        self.pipeline = pipeline
        self.elements = elems

        # Python pad probes: _select_probe picks which decoded frames to keep
        # (by index, or by PTS in GOP-drop mode) and drops the rest before
        # nvvideoconvert; _copy_probe copies the kept RGB frames into the
        # destination tensor.
        BUF = Gst.PadProbeType.BUFFER
        self._probe_ids["select"] = elems["nvvconv"].get_static_pad(
            "sink").add_probe(BUF, self._select_probe, None)
        self._probe_ids["copy"] = elems["capsf"].get_static_pad(
            "src").add_probe(BUF, self._copy_probe, None)

        # Attach the appsrc -> parsebin source. No warmup possible — appsrc
        # has no bytes yet; the first decode primes the NVDEC session and
        # every later decode reuses it.
        self._add_source()

    def _add_source(self) -> None:
        """Create ``appsrc -> parsebin`` and link parsebin into the
        retained nvv4l2decoder. parsebin's src pad is dynamic, so the link
        happens in the pad-added callback."""
        Gst = self._Gst
        appsrc = Gst.ElementFactory.make("appsrc", None)
        parsebin = Gst.ElementFactory.make("parsebin", None)
        if appsrc is None or parsebin is None:
            raise RuntimeError("appsrc/parsebin creation failed")

        # stream-type=0 (GST_APP_STREAM_TYPE_STREAM): push mode, not
        # seekable. The whole container is pushed as one buffer per decode
        # and frames are selected by index. No caps set: parsebin
        # type-finds the container from the bytes.
        appsrc.set_property("stream-type", 0)
        appsrc.set_property("format", Gst.Format.BYTES)
        appsrc.set_property("is-live", False)
        appsrc.set_property("block", True)
        appsrc.set_property("max-bytes", 0)  # unlimited; whole file per push

        self.pipeline.add(appsrc)
        self.pipeline.add(parsebin)
        if not appsrc.link(parsebin):
            raise RuntimeError("appsrc->parsebin link failed")

        BUF = Gst.PadProbeType.BUFFER
        q_in = self.elements["q_in"]

        def _on_parsebin_pad(_pb, pad):
            cap = pad.get_current_caps() or pad.query_caps(None)
            if not cap or cap.get_size() == 0:
                return
            if not cap.get_structure(0).get_name().startswith("video/"):
                return
            # Link into the input queue (head of the warm chain), not the
            # decoder directly — the queue feeds NVDEC.
            sink_pad = q_in.get_static_pad("sink")
            if sink_pad is None or sink_pad.is_linked():
                return
            if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                return
            # The parser GOP-drop probe is needed in PTS mode and in
            # index-mode GOP-drop; in plain index mode (full decode) it is a
            # no-op, so don't attach it — that avoids a Python pad-probe
            # callback per compressed frame (GIL traffic). The pad-added
            # callback fires after PLAYING, so use_pts_mode / gop_drop are
            # already set by _reset_for_decode for this request.
            if self.use_pts_mode or self.gop_drop:
                self._probe_ids["parser"] = pad.add_probe(
                    BUF, self._parser_probe, None)
        parsebin.connect("pad-added", _on_parsebin_pad)

        self.elements["appsrc"] = appsrc
        self.elements["parsebin"] = parsebin
        appsrc.sync_state_with_parent()
        parsebin.sync_state_with_parent()

    def _reset_source(self) -> None:
        """Recreate only ``appsrc -> parsebin`` for a fresh byte stream,
        keeping the nvv4l2decoder (and its NVDEC session) warm.

        The retained decoder chain is flushed to clear the EOS/segment
        left by the previous stream — without a state change — so NVDEC is
        never re-initialised.
        """
        Gst = self._Gst
        # Head of the warm chain is now q_in (queue) → nvdec → ...
        q_in_sink = self.elements["q_in"].get_static_pad("sink")

        # Unlink the old parsebin, then NULL + remove the appsrc/parsebin
        # pair (parsebin cannot be reused for a second stream).
        peer = q_in_sink.get_peer()
        if peer is not None:
            peer.unlink(q_in_sink)
        for key in ("parsebin", "appsrc"):
            e = self.elements.pop(key, None)
            if e is not None:
                e.set_state(Gst.State.NULL)
                self.pipeline.remove(e)

        # Flush the retained chain (q_in -> nvdec -> q_dec -> nvvconv -> ... ->
        # sink) to clear the prior stream's EOS so it accepts the new
        # segment, keeping every element — and the NVDEC session — alive.
        q_in_sink.send_event(Gst.Event.new_flush_start())
        q_in_sink.send_event(Gst.Event.new_flush_stop(True))

        self._add_source()

    # ------------------------------------------------------------------
    def decode(self, req: _DecodeRequest) -> _DecodeResult:
        Gst = self._ensure_gst()
        try:
            self.ensure_pipeline()
        except Exception as e:
            return _DecodeResult(
                job_id=req.job_id, worker_id=self.worker_id,
                error=f"{type(e).__name__}: {e}")

        # A pipeline that already ran a byte stream needs a fresh appsrc ->
        # parsebin source before the next stream. If the codec matches the
        # warm decoder, _reset_source recreates only that pair and flushes
        # the decoder, keeping the NVDEC session warm. If the codec changed
        # (H.264 <-> H.265), fall back to a full NULL-cycle rebuild.
        if self.pipeline_has_streamed:
            codec_changed = (
                bool(req.codec) and bool(self.current_codec)
                and req.codec != self.current_codec
            )
            if codec_changed:
                self._rebuild_all()
            else:
                self._reset_source()
        self.current_codec = req.codec or self.current_codec

        self._reset_for_decode(req)
        if self.debug_pts:
            logger.info(
                "[ds w%d] IN  %s mode: %d targets",
                self.worker_id,
                "PTS" if self.use_pts_mode else "index",
                len(self.target_pts if self.use_pts_mode
                    else self.target_indices))
        # Frame size may differ per stream — clear the sticky dims.
        self.width = 0
        self.height = 0

        bus = self.pipeline.get_bus()
        while bus.pop_filtered(
                Gst.MessageType.EOS | Gst.MessageType.ERROR) is not None:
            pass

        self.pipeline.set_state(Gst.State.PLAYING)

        # Push the whole container as a single buffer, then end the stream.
        # new_wrapped keeps a reference to the bytes — no copy.
        appsrc = self.elements["appsrc"]
        flow = appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(req.data))
        if flow != Gst.FlowReturn.OK:
            return self._finalize_result(
                req, f"appsrc push-buffer returned {flow!r}")
        appsrc.emit("end-of-stream")

        msg = bus.timed_pop_filtered(
            int(req.timeout_sec * 1e9),
            Gst.MessageType.EOS | Gst.MessageType.ERROR)
        error = ""
        if msg is None:
            error = f"timeout after {req.timeout_sec}s"
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            error = f"{err.message} ({dbg})"

        # Mark streamed so the next decode recreates the source first.
        self.pipeline_has_streamed = True
        return self._finalize_result(req, error)

    def shutdown(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.set_state(self._Gst.State.NULL)
        self.pipeline = None
        self.elements = {}


# ----------------------------------------------------------------------
# RTSP / persistent-stream worker — pipeline stays in PLAYING forever
# ----------------------------------------------------------------------
class _StreamWorkerState(_BaseWorkerState):
    def __init__(self, uri: str, drop_interval: int):
        super().__init__(worker_id=0, drop_interval=drop_interval)
        self.uri = uri
        # Live streams cannot accept EOS — use a threading.Event in the
        # worker process to wake the request handler when the count probe
        # has captured ``max_frames``.
        import threading
        self._done_event = threading.Event()

    def start(self) -> None:
        Gst = self._ensure_gst()
        self._ensure_cudart()

        elems = {
            "uridec":  Gst.ElementFactory.make("uridecodebin",   None),
            "nvvconv": Gst.ElementFactory.make("nvvideoconvert", None),
            "capsf":   Gst.ElementFactory.make("capsfilter",     None),
            "sink":    Gst.ElementFactory.make("fakesink",       None),
        }
        missing = [k for k, v in elems.items() if v is None]
        if missing:
            raise RuntimeError(f"stream element creation failed: {missing}")

        elems["uridec"].set_property("uri", self.uri)
        elems["nvvconv"].set_property("nvbuf-memory-type", 2)
        elems["sink"].set_property("sync", False)
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGB")
        elems["capsf"].set_property("caps", caps)

        pipeline = Gst.Pipeline.new(None)
        for e in elems.values():
            pipeline.add(e)
        if not (elems["nvvconv"].link(elems["capsf"])
                and elems["capsf"].link(elems["sink"])):
            raise RuntimeError("stream downstream link failed")

        def _on_uridec_pad(_dec, pad):
            cap = pad.query_caps(None)
            if not cap or cap.get_size() == 0:
                return
            name = cap.get_structure(0).get_name()
            if not name.startswith("video/"):
                return
            sink_pad = elems["nvvconv"].get_static_pad("sink")
            if sink_pad and not sink_pad.is_linked():
                pad.link(sink_pad)
        elems["uridec"].connect("pad-added", _on_uridec_pad)

        BUF = Gst.PadProbeType.BUFFER
        elems["nvvconv"].get_static_pad("sink").add_probe(
            BUF, self._select_probe, None)
        elems["capsf"].get_static_pad("src").add_probe(
            BUF, self._copy_probe, None)

        self.pipeline = pipeline
        self.elements = elems

        pipeline.set_state(Gst.State.PLAYING)

    def decode_segment(self, req: _DecodeRequest) -> _DecodeResult:
        if self.pipeline is None:
            return _DecodeResult(
                job_id=req.job_id, worker_id=self.worker_id,
                error="stream pipeline not running")

        self._reset_for_decode(req)

        # Block until the copy probe signals (via threading.Event) or we
        # time out. Pipeline keeps running; an upstream error is caught
        # by polling the bus non-blocking after the wait.
        signaled = self._done_event.wait(req.timeout_sec)

        Gst = self._Gst
        bus = self.pipeline.get_bus()
        error = ""
        if not signaled:
            error = f"timeout after {req.timeout_sec}s"
        msg = bus.pop_filtered(Gst.MessageType.ERROR)
        if msg is not None:
            err, dbg = msg.parse_error()
            error = f"{err.message} ({dbg})"

        return self._finalize_result(req, error)

    def shutdown(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.set_state(self._Gst.State.NULL)
        self.pipeline = None
        self.elements = {}


# ----------------------------------------------------------------------
# GstBuffer → NvBufSurface accessor
# ----------------------------------------------------------------------
_NVBUF_SURFACE_HEAD_SZ = ctypes.sizeof(_NvBufSurface)


def _read_nvbuf_surface_first(buf, Gst):
    """Read pitch + dataPtr + width + height from ``surfaceList[0]``.

    ``buf.map(GST_MAP_READ)`` exposes the GstBuffer's memory as a Python
    bytes view; for NVMM buffers that view is the raw NvBufSurface
    struct (numFilled, surfaceList, …). The struct contains a
    ``surfaceList`` pointer pointing to NvBufSurfaceParams in the same
    allocation, which we then dereference via ctypes to read the GPU
    pointer (``dataPtr``) and pitch.

    Returns ``(data_ptr, pitch, width, height)``. All zero on failure.
    """
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return 0, 0, 0, 0
    try:
        if mapinfo.size < _NVBUF_SURFACE_HEAD_SZ:
            return 0, 0, 0, 0
        # mapinfo.data is a bytes-like view; copy out the head of the
        # NvBufSurface struct so we can parse it. The pointer values
        # inside the copy still reference the original NVMM allocation.
        head = bytes(mapinfo.data[:_NVBUF_SURFACE_HEAD_SZ])
        surf = _NvBufSurface.from_buffer_copy(head)
        if surf.numFilled == 0 or not surf.surfaceList:
            return 0, 0, 0, 0
        p = surf.surfaceList[0]
        data_ptr = int(p.dataPtr or 0)
        if not data_ptr and p.cudaBuffer:
            # NvBufSurfaceCudaBuffer layout: void* basePtr; void* dataPtr; …
            # The second pointer is the page-aligned image data.
            data_ptr = int(ctypes.c_void_p.from_address(
                int(p.cudaBuffer) + 8).value or 0)
        return data_ptr, int(p.pitch), int(p.width), int(p.height)
    finally:
        buf.unmap(mapinfo)


# ======================================================================
# Public Pool / Stream API
# ======================================================================
@dataclass
class DecodeFrames:
    """Result of a single decode call. ``frames`` is a CUDA tensor
    ``(n_kept, H, W, 3)`` uint8 in the same process; access directly."""
    frames: torch.Tensor | None
    n_kept: int
    n_total: int
    fps: float
    error: str = ""


def _file_worker_loop(worker_id: int,
                      drop_interval: int,
                      req_q,
                      res_q,
                      closed_flag: "list[bool]") -> None:
    state = _FileWorkerState(worker_id, drop_interval)
    # The appsrc pipeline is built lazily on the first decode — there is
    # nothing to warm up without bytes in hand.
    try:
        while not closed_flag[0]:
            req: _DecodeRequest | None = req_q.get()
            if req is None:
                break
            res = state.decode(req)
            res_q.put(res)
    finally:
        state.shutdown()


def _stream_worker_loop(uri: str,
                        drop_interval: int,
                        req_q,
                        res_q,
                        closed_flag: "list[bool]") -> None:
    state = _StreamWorkerState(uri, drop_interval)
    try:
        state.start()
    except Exception as e:
        res_q.put(_DecodeResult(
            job_id=-1, worker_id=0,
            error=f"stream start failed: {e}"))
        return
    try:
        while not closed_flag[0]:
            req: _DecodeRequest | None = req_q.get()
            if req is None:
                break
            res = state.decode_segment(req)
            res_q.put(res)
    finally:
        state.shutdown()


class DecodePool:
    """Pool of N file-decode pipelines on N daemon threads, sharing one
    CUDA context. Probe GIL transitions are negligible at our probe call
    rate, and a single CUDA context lets the driver pipeline NVDEC
    sessions efficiently across pool slots.

    Frames returned by :meth:`decode` are CUDA tensors in this same
    process — no IPC handle, no D2H/H2D round-trip; the caller uses the
    tensor directly.
    """

    def __init__(self,
                 num_workers: int = 8,
                 drop_interval: int = 0) -> None:
        import queue as _queue
        import threading

        # Force CUDA context init in the main thread before workers build
        # pipelines, so nvv4l2decoder/nvvideoconvert pick up the same
        # primary context PyTorch uses.
        if torch.cuda.is_available():
            torch.cuda.init()

        self._req_q: "_queue.Queue" = _queue.Queue()
        self._res_q: "_queue.Queue" = _queue.Queue()
        self._closed_flag = [False]
        self._workers: list[threading.Thread] = []
        for i in range(num_workers):
            t = threading.Thread(
                target=_file_worker_loop,
                args=(i, drop_interval,
                      self._req_q, self._res_q, self._closed_flag),
                daemon=True,
                name=f"ds-decode-thread-{i}",
            )
            t.start()
            self._workers.append(t)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._results: dict[int, _DecodeResult] = {}
        self._next_id = 0
        self._closed = False
        self._collector = threading.Thread(
            target=self._collect_loop, daemon=True,
            name="ds-decode-pool-collector")
        self._collector.start()

    def decode(self,
               data: bytes,
               *,
               target_indices: list[int] | None = None,
               target_pts_ns: list[int] | None = None,
               codec: str = "",
               max_frames: int = 8,
               gop_drop: bool = False,
               fps: float = 0.0,
               timeout_sec: float = 30.0) -> DecodeFrames:
        """Decode ``data`` (raw container bytes) on a pool worker.

        Selection mode is chosen by which target list is given (pass one):

        * ``target_indices`` — **index mode**: exactly those frames are kept
          (1:1 with the indices). By default the whole stream is decoded; set
          ``gop_drop=True`` (and pass ``fps``) to additionally drop whole GOPs
          containing no target at the parser, so NVDEC only decodes the GOPs
          that matter — exact, and a big win for sparse sampling of long
          videos (~no gain when the sample is as dense as the GOP count).
        * ``target_pts_ns`` — **PTS mode (GOP-drop)**: GOPs containing none
          of the targets are dropped at the parser. Selection is by PTS and
          not guaranteed 1:1 when a stream's PTS don't align with the targets.

        ``codec`` (e.g. ``"h264"``/``"hevc"``) is an optional hint that lets
        a worker keep its NVDEC session warm across same-codec streams and
        rebuild only on a codec change. Omitting it disables that check.
        """
        if self._closed:
            raise RuntimeError("DecodePool is closed")
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
        self._req_q.put(_DecodeRequest(
            job_id=job_id,
            data=data,
            codec=codec,
            target_pts_ns=tuple(target_pts_ns) if target_pts_ns else (),
            target_indices=tuple(target_indices) if target_indices else (),
            use_pts_mode=target_pts_ns is not None,
            max_frames=max_frames,
            gop_drop=gop_drop,
            fps=fps,
            timeout_sec=timeout_sec,
        ))
        with self._cv:
            while job_id not in self._results and not self._closed:
                self._cv.wait()
            if self._closed and job_id not in self._results:
                raise RuntimeError(
                    "DecodePool was closed before result")
            res = self._results.pop(job_id)
        return _to_decode_frames(res)

    def _collect_loop(self) -> None:
        import queue as _queue
        while not self._closed:
            try:
                res = self._res_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            with self._cv:
                self._results[res.job_id] = res
                self._cv.notify_all()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closed_flag[0] = True
        for _ in self._workers:
            self._req_q.put(None)
        for t in self._workers:
            t.join(timeout=5)
        with self._cv:
            self._cv.notify_all()


class StreamHandle:
    """One persistent RTSP/URI pipeline on a daemon thread. Shares the
    process-wide CUDA context with any active :class:`DecodePool`."""

    def __init__(self, uri: str, drop_interval: int = 0) -> None:
        import queue as _queue
        import threading

        if torch.cuda.is_available():
            torch.cuda.init()

        self._req_q: "_queue.Queue" = _queue.Queue()
        self._res_q: "_queue.Queue" = _queue.Queue()
        self._closed_flag = [False]
        self._uri = uri
        self._next_id = 0
        self._closed = False
        self._worker = threading.Thread(
            target=_stream_worker_loop,
            args=(uri, drop_interval,
                  self._req_q, self._res_q, self._closed_flag),
            daemon=True,
            name=f"ds-stream-thread-{uri[:32]}",
        )
        self._worker.start()

    def decode_segment(self,
                       *,
                       target_indices: list[int] | None = None,
                       max_frames: int = 8,
                       timeout_sec: float = 30.0) -> DecodeFrames:
        if self._closed:
            raise RuntimeError("StreamHandle is closed")
        job_id = self._next_id
        self._next_id += 1
        self._req_q.put(_DecodeRequest(
            job_id=job_id,
            uri=self._uri,
            target_indices=tuple(target_indices) if target_indices else (),
            use_pts_mode=False,
            max_frames=max_frames,
            timeout_sec=timeout_sec,
        ))
        res: _DecodeResult = self._res_q.get()
        return _to_decode_frames(res)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closed_flag[0] = True
        self._req_q.put(None)
        self._worker.join(timeout=5)


def _to_decode_frames(res: _DecodeResult) -> DecodeFrames:
    return DecodeFrames(
        frames=res.frames,
        n_kept=res.n_kept,
        n_total=res.n_total,
        fps=res.fps,
        error=res.error,
    )


# ======================================================================
# Metadata probe — GStreamer-only, no frame decode, no external library
# ======================================================================
def _caps_name_to_codec(name: str) -> str:
    """Map a GStreamer caps name to a stable codec key for the warm-reuse
    hint. Consistency across calls matters, not matching any other tool."""
    if name.startswith("video/x-h264"):
        return "h264"
    if name.startswith("video/x-h265") or "hevc" in name:
        return "hevc"
    # Fall back to the bare media subtype (e.g. "video/x-vp9" -> "x-vp9").
    return name.split("/", 1)[-1] if "/" in name else name


def probe_metadata(
    data: bytes,
    timeout_sec: float = 10.0,
) -> tuple[int, float, float, int, int, str]:
    """Read container metadata from raw bytes using GStreamer only.

    Returns ``(frame_count, fps, duration_sec, width, height, codec)``.
    No frames are decoded: the pipeline is ``appsrc -> parsebin ->
    fakesink`` prerolled to PAUSED, so ``parsebin`` only demuxes the
    container and parses the elementary-stream headers. ``fps``, ``width``,
    ``height`` and ``codec`` come from the parsed caps; ``duration`` from a
    pipeline duration query; ``frame_count`` is derived as
    ``round(duration * fps)``. Unknown numeric fields are returned as 0 and
    ``codec`` as ``""``.

    Uses only core GStreamer elements — the DeepStream plugins are not
    required for a probe.
    """
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # type: ignore
    Gst.init(None)

    pipeline = Gst.Pipeline.new(None)
    appsrc = Gst.ElementFactory.make("appsrc", None)
    parsebin = Gst.ElementFactory.make("parsebin", None)
    sink = Gst.ElementFactory.make("fakesink", None)
    if not (appsrc and parsebin and sink):
        raise RuntimeError("probe: GStreamer element creation failed")

    appsrc.set_property("stream-type", 0)  # STREAM (push)
    appsrc.set_property("format", Gst.Format.BYTES)
    appsrc.set_property("is-live", False)
    appsrc.set_property("block", False)
    appsrc.set_property("max-bytes", 0)  # unlimited; whole file in one push
    sink.set_property("sync", False)

    for e in (appsrc, parsebin, sink):
        pipeline.add(e)
    if not appsrc.link(parsebin):
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("probe: appsrc->parsebin link failed")

    def _on_pad(_pb, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return
        if not caps.get_structure(0).get_name().startswith("video/"):
            return
        sink_pad = sink.get_static_pad("sink")
        if sink_pad is not None and not sink_pad.is_linked():
            pad.link(sink_pad)
    parsebin.connect("pad-added", _on_pad)

    try:
        pipeline.set_state(Gst.State.PAUSED)
        # Push the whole container + EOS so parsebin can parse the headers
        # (incl. a trailing MP4 moov) and reach preroll.
        appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(data))
        appsrc.emit("end-of-stream")
        # Block until preroll (state change to PAUSED) completes.
        pipeline.get_state(int(timeout_sec * Gst.SECOND))

        # Read the *negotiated* caps after preroll — at pad-added time only
        # the media type is known; framerate/width/height are filled in once
        # the parser has negotiated downstream.
        fps = 0.0
        width = height = 0
        codec = ""
        sink_caps = sink.get_static_pad("sink").get_current_caps()
        if sink_caps is not None and sink_caps.get_size() > 0:
            s = sink_caps.get_structure(0)
            codec = _caps_name_to_codec(s.get_name())
            ok, num, den = s.get_fraction("framerate")
            if ok and den:
                fps = num / den
            okw, w = s.get_int("width")
            okh, h = s.get_int("height")
            width = w if okw else 0
            height = h if okh else 0

        dur_ok, dur_ns = pipeline.query_duration(Gst.Format.TIME)
        duration = (dur_ns / 1e9) if (dur_ok and dur_ns > 0) else 0.0
        frame_count = round(duration * fps) if (duration > 0 and fps > 0) else 0
        return frame_count, fps, duration, width, height, codec
    finally:
        pipeline.set_state(Gst.State.NULL)
