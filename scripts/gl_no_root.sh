#!/usr/bin/env bash
# Install libglvnd's EGL dispatch libraries into $HOME. No root, no container.
#
#   bash scripts/gl_no_root.sh && . ~/.mavrl_gl_env && python -m mavrl.glcheck
#
# `.` rather than `source`: the file has to be readable by dash, which is
# /bin/sh on Debian and Ubuntu and does not have bash's `source` builtin.
#
# For the case glcheck reports as "the NVIDIA driver's EGL is present -- only
# libglvnd's dispatch layer is missing": libEGL_nvidia.so.0 loads fine, but the
# libEGL.so.1 shim that routes calls to it was never installed, so PyOpenGL
# finds nothing and hands MuJoCo None.
#
# `apt-get download` and `dpkg-deb -x` are both unprivileged -- the packages are
# unpacked into a directory you own rather than installed system-wide. Nothing
# outside $PREFIX is touched, so removing that directory undoes all of this.

set -euo pipefail

PREFIX="${MAVRL_GL_PREFIX:-$HOME/.local/gl}"
ENVFILE="$HOME/.mavrl_gl_env"
ARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)"
LIBDIR="$PREFIX/usr/lib/$ARCH"

echo "==> unpacking into $PREFIX"
mkdir -p "$PREFIX/debs"
cd "$PREFIX/debs"

# libegl1 carries libEGL.so.1, libopengl0 carries libOpenGL.so.0. libglvnd0 is
# already present on this box (libGLX.so.0 loads) but is listed anyway, since
# the same script has to work where it is not.
if ! apt-get download libegl1 libopengl0 libglvnd0 2>/dev/null; then
    echo "!! apt-get download failed."
    echo "   Usually the apt package lists are missing or the host is offline."
    echo "   Fall back to the container: docker build -t mavrl -f docker/Dockerfile ."
    exit 1
fi

for deb in *.deb; do
    echo "    $deb"
    dpkg-deb -x "$deb" "$PREFIX"
done

if [ ! -e "$LIBDIR/libEGL.so.1" ]; then
    echo "!! libEGL.so.1 did not land in $LIBDIR -- look under $PREFIX and adjust"
    exit 1
fi

# ctypes.util.find_library('EGL') -- which is what PyOpenGL calls -- shells out
# to `ld -lEGL` when the ldconfig cache misses, and `ld` wants the unversioned
# `libEGL.so` development symlink. The runtime package ships only libEGL.so.1,
# so the symlink has to be made here or find_library keeps returning None and
# nothing changes.
cd "$LIBDIR"
for stem in EGL OpenGL GLX GLdispatch; do
    for real in lib$stem.so.[0-9]*; do
        [ -e "$real" ] && ln -sf "$real" "lib$stem.so"
    done
done

cat > "$ENVFILE" <<EOF
# Written by scripts/gl_no_root.sh. Source before running anything that renders.
export LD_LIBRARY_PATH="$LIBDIR:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$LIBDIR:\${LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
EOF

echo
echo "==> done. Now, in this same shell:"
echo "      . $ENVFILE"
echo "      python -m mavrl.glcheck"
echo
echo "    Add '. $ENVFILE' to ~/.bashrc to make it stick."
