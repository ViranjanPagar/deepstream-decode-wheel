#!/usr/bin/env bash
# Build the deepstream-decode wheel.
#
# Layout it produces (flat — all libs in one directory):
#
#   src/deepstream_decode/_libs/
#       libnvbufsurface.so
#       libnvbufsurftransform.so
#       libnvbuf_fdmap.so
#       libnvds_meta.so
#       libnvdsbufferpool.so
#       libnvdsgst_helper.so
#       libnvdsgst_meta.so
#       libgstnvdsseimeta.so
#       libgstnvcustomhelper.so
#       libnvv4l2.so                  ← from /work/libnvv4l2.so (patched, temporary)
#       libcuvidv4l2.so
#       libv4l2.so.0                  ← symlink → libnvv4l2.so
#       libgstnvvideo4linux2.so       (GStreamer plugin)
#       libgstnvvideoconvert.so       (GStreamer plugin)
#       libcuvidv4l2_plugin.so        (libv4l plugin)
#
# All .so files get DT_RUNPATH=$ORIGIN so inter-lib NEEDED entries resolve
# to siblings.
#
# Inputs (defaults shown):
#   DS_TBZ2              = (auto-resolved: ./<tarball> next to this script;
#                           if not present, wget downloads from NGC.)
#   DS_VERSION           = 9.0.0
#   PATCHED_LIBNVV4L2    = <recipe-dir>/libnvv4l2.so  (temporary, until the
#                          tbz2 ships a libnvv4l2.so with LIBV4L2_PLUGIN_DIR
#                          support)
#
# Single-arg overrides:
#   ./build_wheel.sh --ds-version=9.1.0
#   ./build_wheel.sh --ds-tbz2=/path/file.tbz2
#   ./build_wheel.sh --ds-src=/path/to/extracted_tree
#   ./build_wheel.sh --no-download                # fail if tbz2 not found locally
#
# How to fetch the DS tbz2 from NVIDIA (license-gated, not redistributable):
#   1. Auto-download: this script tries
#        https://api.ngc.nvidia.com/v2/resources/nvidia/deepstream/versions/<MM>/files/deepstream_sdk_v<VERSION>_x86_64.tbz2
#      Some versions need an NGC API key — if `wget` returns 401, manual
#      download from developer.nvidia.com is the fallback.
#   2. Manual download: https://developer.nvidia.com/deepstream-getting-started
#      → pick the version → Linux x86_64 → Tar Package. Place next to this
#      script as deepstream_sdk_v<VERSION>_x86_64.tbz2.

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DS_VERSION="${DS_VERSION:-9.0.0}"
DS_TBZ2="${DS_TBZ2:-}"
DS_SRC_OVERRIDE=""
NO_DOWNLOAD=0
FORCE_DOWNLOAD=0
PATCHED_LIBNVV4L2="${PATCHED_LIBNVV4L2:-$RECIPE_DIR/libnvv4l2.so}"

for arg in "$@"; do
    case "$arg" in
        --ds-version=*)   DS_VERSION="${arg#--ds-version=}" ;;
        --ds-tbz2=*)      DS_TBZ2="${arg#--ds-tbz2=}" ;;
        --ds-src=*)       DS_SRC_OVERRIDE="${arg#--ds-src=}" ;;
        --no-download)    NO_DOWNLOAD=1 ;;
        --force-download) FORCE_DOWNLOAD=1 ;;
        -h|--help)
            sed -n 's/^# \{0,1\}//p; /^[^#]/q' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

DS_MAJOR_MINOR="${DS_VERSION%.*}"
TARBALL_NAME="deepstream_sdk_v${DS_VERSION}_x86_64.tbz2"
# NGC stores DeepStream versions as MAJOR.MINOR (e.g., "9.0"), not full
# SemVer. The recipes endpoint (NGC's SDK-asset namespace) returns a 302
# redirect to a signed CloudFront URL on xfiles.ngc.nvidia.com.
TARBALL_URL="https://api.ngc.nvidia.com/v2/recipes/nvidia/deepstream/versions/${DS_MAJOR_MINOR}/files/${TARBALL_NAME}"
EXTRACT_DIR="$RECIPE_DIR/deepstream_sdk_v${DS_VERSION}_x86_64"

PKG_DIR="$RECIPE_DIR/src/deepstream_decode"
LIBS_DIR="$PKG_DIR/_libs"
DIST_DIR="$RECIPE_DIR/dist"

echo "==> Recipe:           $RECIPE_DIR"
echo "==> DS_VERSION:       $DS_VERSION"
echo "==> Libs install at:  $LIBS_DIR (flat)"
echo "==> Patched libnvv4l2.so override: $PATCHED_LIBNVV4L2"

# ─── tool + input checks ─────────────────────────────────────────────────
for tool in patchelf tar; do
    command -v "$tool" >/dev/null 2>&1 \
        || { echo "ERROR: '$tool' required (apt install $tool)" >&2; exit 1; }
done
command -v objdump >/dev/null 2>&1 || command -v readelf >/dev/null 2>&1 \
    || { echo "ERROR: install binutils" >&2; exit 1; }

[ -f "$PATCHED_LIBNVV4L2" ] \
    || { echo "ERROR: PATCHED_LIBNVV4L2=$PATCHED_LIBNVV4L2 not found." >&2; exit 1; }
if ! strings "$PATCHED_LIBNVV4L2" 2>/dev/null | grep -q '^LIBV4L2_PLUGIN_DIR$'; then
    echo "ERROR: $PATCHED_LIBNVV4L2 doesn't have LIBV4L2_PLUGIN_DIR support." >&2
    echo "       The wheel won't be able to direct it at the bundled libv4l plugin." >&2
    exit 1
fi

# ─── 1. Acquire DS source tree ───────────────────────────────────────────
DS_SRC=""
if [ -n "$DS_SRC_OVERRIDE" ]; then
    DS_SRC="$DS_SRC_OVERRIDE"
    echo "==> [1/5] DS source (user-provided): $DS_SRC"
else
    SOURCE_TBZ2=""
    if [ "$FORCE_DOWNLOAD" -eq 0 ]; then
        if [ -n "$DS_TBZ2" ] && [ -f "$DS_TBZ2" ]; then
            SOURCE_TBZ2="$DS_TBZ2"
        elif [ -f "$RECIPE_DIR/$TARBALL_NAME" ]; then
            SOURCE_TBZ2="$RECIPE_DIR/$TARBALL_NAME"
        fi
    fi

    if [ -n "$SOURCE_TBZ2" ]; then
        echo "==> [1/5] Using local tarball: $SOURCE_TBZ2"
    else
        # No local copy — try NGC download.
        if [ "$NO_DOWNLOAD" -eq 1 ]; then
            echo "ERROR: $TARBALL_NAME not found and --no-download set." >&2
            echo "Place the tarball next to this script, or set DS_TBZ2=<path>," >&2
            echo "or pass --ds-src=<extracted_tree>. See header for fetch instructions." >&2
            exit 1
        fi
        SOURCE_TBZ2="$RECIPE_DIR/$TARBALL_NAME"
        echo "==> [1/5] Downloading $TARBALL_NAME from NGC"
        echo "         $TARBALL_URL"
        command -v wget >/dev/null 2>&1 \
            || { echo "ERROR: wget required for download (apt install wget)." >&2; exit 1; }

        # NGC API key is taken from $NGC_API_KEY if set. NGC's public-resource
        # endpoint accepts unauthenticated requests for some assets but
        # gated ones (most SDK tarballs) require an API key as a Bearer
        # token. To use: `source your_env_with_NGC_API_KEY && ./build_wheel.sh`.
        WGET_AUTH=()
        if [ -n "${NGC_API_KEY:-}" ]; then
            WGET_AUTH=(--header="Authorization: Bearer $NGC_API_KEY")
            echo "         (authenticated — using NGC_API_KEY)"
        else
            echo "         (unauthenticated — set NGC_API_KEY if download fails 401)"
        fi

        wget "${WGET_AUTH[@]}" --content-disposition "$TARBALL_URL" -O "$SOURCE_TBZ2" \
            || { echo "ERROR: download failed. If you got HTTP 401, set NGC_API_KEY (https://ngc.nvidia.com → Setup → API Key). Otherwise the version may not be on NGC — see header for the manual developer-portal fetch path." >&2; rm -f "$SOURCE_TBZ2"; exit 1; }
    fi

    # Reuse an existing extract dir if it has libnvbufsurface.so.
    if [ -d "$EXTRACT_DIR" ] && find "$EXTRACT_DIR" -name libnvbufsurface.so -print -quit 2>/dev/null | grep -q .; then
        echo "         Already extracted at $EXTRACT_DIR"
    else
        echo "         Extracting (takes a few minutes)..."
        rm -rf "$EXTRACT_DIR"; mkdir -p "$EXTRACT_DIR"
        tar -xjf "$SOURCE_TBZ2" -C "$EXTRACT_DIR"
    fi

    DS_SRC=$(find "$EXTRACT_DIR" \
        -path "*/opt/nvidia/deepstream/*/lib/libnvbufsurface.so" \
        -print -quit 2>/dev/null \
        | xargs -r -I{} dirname {} | xargs -r -I{} dirname {})
    [ -n "$DS_SRC" ] || { echo "ERROR: libnvbufsurface.so not found under $EXTRACT_DIR" >&2; exit 1; }
fi
echo "    DS source root: $DS_SRC"

# ─── 2. Stage .so files into _libs/ ──────────────────────────────────────
# Layout: everything flat EXCEPT the libv4l plugin, which lives in a
# dedicated sub-dir. libnvv4l2.so's plugin scanner tries dlopen+dlsym on
# every .so in its plugin dir — putting non-plugin libs there triggers
# noisy "dlsym failed" warnings. The v4l_plugins/ sub-dir contains ONLY
# real libv4l plugins, so the scan is quiet.
#
# GStreamer's plugin scanner is more forgiving (checks for a magic
# descriptor field, not a specific symbol), so the GStreamer plugins
# can stay alongside the main libs without complaints.
echo "==> [2/5] Staging .so files (flat + v4l_plugins/ sub-dir)"
rm -rf "$LIBS_DIR"; mkdir -p "$LIBS_DIR/v4l_plugins"

# 10 SDK libs that always come from the tbz2 (libnvv4l2.so is the 11th, see below).
LIBS_FROM_SDK_ROOT=(
    libnvbufsurface.so
    libnvbufsurftransform.so
    libnvbuf_fdmap.so
    libnvds_meta.so
    libnvdsbufferpool.so
    libnvdsgst_helper.so
    libnvdsgst_meta.so
    libgstnvdsseimeta.so
    libgstnvcustomhelper.so
    libcuvidv4l2.so
)
for f in "${LIBS_FROM_SDK_ROOT[@]}"; do
    SRC="$DS_SRC/lib/$f"
    [ -f "$SRC" ] || { echo "ERROR: missing in SDK: $SRC" >&2; exit 1; }
    cp -L "$SRC" "$LIBS_DIR/$f"
done

# libnvv4l2.so — TEMPORARY override from /work/libnvv4l2.so.
# When the SDK tbz2 starts shipping the patched version, swap this line
# for `cp -L "$DS_SRC/lib/libnvv4l2.so" "$LIBS_DIR/"`.
echo "    libnvv4l2.so ← $PATCHED_LIBNVV4L2  (override)"
cp -L "$PATCHED_LIBNVV4L2" "$LIBS_DIR/libnvv4l2.so"

# GStreamer plugins — staged flat into _libs/ alongside main libs.
cp -L "$DS_SRC/lib/gst-plugins/libgstnvvideo4linux2.so"    "$LIBS_DIR/"
cp -L "$DS_SRC/lib/gst-plugins/libgstnvvideoconvert.so"    "$LIBS_DIR/"

# libv4l plugin — staged into the dedicated v4l_plugins/ sub-dir so
# libnvv4l2.so's scanner only sees real plugins (no dlsym noise).
cp -L "$DS_SRC/lib/libv4l/plugins/libcuvidv4l2_plugin.so"  "$LIBS_DIR/v4l_plugins/"

# libv4l2.so.0 SONAME alias — symlink to libnvv4l2.so. libgstnvvideo4linux2.so
# does dlopen("libv4l2.so.0") at runtime; with both files in the same dir
# and $ORIGIN as the RPATH, the symlink resolves correctly.
ln -sf libnvv4l2.so "$LIBS_DIR/libv4l2.so.0"

# ─── 3. Patch DT_RUNPATH ─────────────────────────────────────────────────
# Flat _libs/ files: $ORIGIN (siblings)
# v4l_plugins/*.so:  $ORIGIN/.. (main libs are one level up)
echo "==> [3/5] Patching DT_RUNPATH on bundled .so files"
patchelf_set() {
    patchelf --remove-rpath "$1" 2>/dev/null || true
    patchelf --force-rpath --set-rpath "$2" "$1"
}
for so in "$LIBS_DIR"/*.so; do
    [ -L "$so" ] && continue   # skip symlinks (libv4l2.so.0)
    [ -f "$so" ] || continue
    patchelf_set "$so" '$ORIGIN'
done
for so in "$LIBS_DIR"/v4l_plugins/*.so; do
    [ -f "$so" ] || continue
    patchelf_set "$so" '$ORIGIN/..'
done

# ─── 4. Detect CUDA major + glibc floor for the wheel filename ───────────
# CUDA major: from the NEEDED entry of libnppig.so.<MAJOR> in
# libnvbufsurftransform.so. Tagged into the version's local segment
# (e.g., 9.0.0+cuda13).
NPP_LIB="$LIBS_DIR/libnvbufsurftransform.so"
if command -v objdump >/dev/null 2>&1; then
    NPP_SONAME=$(objdump -p "$NPP_LIB" 2>/dev/null \
        | awk '/NEEDED.*libnppig/ {print $2; exit}')
else
    NPP_SONAME=$(readelf -d "$NPP_LIB" 2>/dev/null \
        | awk '/NEEDED.*libnppig/ {gsub(/[][]/, "", $NF); print $NF; exit}')
fi
CUDA_MAJOR=$(echo "$NPP_SONAME" | sed -n 's/.*\.so\.\([0-9]\+\).*/\1/p')
[ -n "$CUDA_MAJOR" ] || { echo "ERROR: couldn't derive CUDA major from '$NPP_SONAME'" >&2; exit 1; }

# glibc floor: scan all bundled .so files and take the MAX GLIBC version
# from their symbol-version requirements. That's the lowest glibc this
# wheel can possibly support — claim anything lower and pip would happily
# install on a host where the libs can't actually load.
#
# Output of `readelf -V` includes lines like "Name: GLIBC_2.34"; we pull
# every match, version-sort, take the highest, and reformat 2.34 → 2_34
# to match PEP 600's manylinux tag syntax.
GLIBC_FLOOR=$(
    {
        for so in "$LIBS_DIR"/*.so "$LIBS_DIR"/v4l_plugins/*.so; do
            [ -f "$so" ] || continue
            readelf -V "$so" 2>/dev/null \
                | grep -oE "GLIBC_[0-9]+\\.[0-9]+"
        done
    } | sort -uV | tail -1 | sed 's/^GLIBC_//' | tr '.' '_'
)
[ -n "$GLIBC_FLOOR" ] || { echo "ERROR: couldn't derive glibc floor from bundled libs" >&2; exit 1; }
echo "==> [4/5] CUDA major: $CUDA_MAJOR (from $NPP_SONAME)"
echo "         glibc floor: $GLIBC_FLOOR (max symbol-version across all bundled .so)"

# ─── 5. Stamp version + build wheel ──────────────────────────────────────
cat > "$PKG_DIR/_version.py" <<EOF
# SPDX-License-Identifier: Apache-2.0
# Stamped by build_wheel.sh.
__version__ = "$DS_VERSION"
EOF

echo "==> [5/5] Building wheel"
mkdir -p "$DIST_DIR"
(
    cd "$RECIPE_DIR"
    rm -rf build *.egg-info src/*.egg-info
    if command -v uv >/dev/null 2>&1; then
        uv build --wheel --out-dir "$DIST_DIR"
    else
        python3 -m build --wheel --outdir "$DIST_DIR"
    fi
)
BUILT_WHEEL=$(ls -t "$DIST_DIR"/deepstream_decode-*.whl 2>/dev/null | head -n1)
[ -n "$BUILT_WHEEL" ] || { echo "ERROR: no wheel produced" >&2; exit 1; }

BUILT_BASE=$(basename "$BUILT_WHEEL")
NEW_NAME=$(echo "$BUILT_BASE" \
    | sed -E "s/^(deepstream_decode)-([0-9.]+)-py3-none-any\\.whl$/\\1-\\2+cuda${CUDA_MAJOR}-py3-none-manylinux_${GLIBC_FLOOR}_x86_64.whl/")
if [ "$NEW_NAME" != "$BUILT_BASE" ]; then
    mv "$BUILT_WHEEL" "$DIST_DIR/$NEW_NAME"
    BUILT_WHEEL="$DIST_DIR/$NEW_NAME"
fi

SZ=$(du -h "$BUILT_WHEEL" | awk '{print $1}')
echo
echo "Built: $BUILT_WHEEL ($SZ)"
echo
echo "Install:"
echo "  apt install gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} \\"
echo "              gstreamer1.0-libav python3-gi python3-gst-1.0 libv4l-0 \\"
echo "              cuda-libraries-13-0"
echo "  pip install '$BUILT_WHEEL'"
echo
echo "Verify (Python consumer view):"
echo "  python3 -c 'import deepstream_decode; print(deepstream_decode.lib_dir())'"
