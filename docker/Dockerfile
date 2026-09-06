# Headless GPU rendering for MuJoCo, via EGL.
#
# The image supplies the CUDA runtime and the GL *loader*; the driver itself is
# injected at run time by the NVIDIA container toolkit. That split is the whole
# reason headless rendering fails in containers -- see NVIDIA_DRIVER_CAPABILITIES
# below, which is the one line most CUDA Dockerfiles are missing.
#
#   docker build -t mavrl -f docker/Dockerfile .
#   docker run --rm --gpus all mavrl python -m mavrl.glcheck
#
# Build from the repo root, not from docker/: the COPY below wants the whole
# tree, including the IMAV arena under "Files(3)".

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# `graphics` is the load-bearing word. The default capability set is
# `compute,utility`, which mounts CUDA and nvidia-smi but *not* libEGL_nvidia --
# so libglvnd falls through to Mesa and every frame is rasterized on the CPU,
# silently and about fifty times too slowly. `--gpus all` alone does not fix it.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libglvnd0 libgl1 libegl1 libgles2 libglx0 \
        libglib2.0-0 libxext6 libx11-6 \
        libosmesa6 \
        git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# libglvnd finds NVIDIA's EGL through this ICD. The toolkit usually mounts it,
# but not on every host/toolkit version, and when it is absent the failure looks
# exactly like a working install that happens to be slow. Writing it costs
# nothing and is idempotent -- a mounted copy overwrites this one.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
      '{' \
      '    "file_format_version" : "1.0.0",' \
      '    "ICD" : {' \
      '        "library_path" : "libEGL_nvidia.so.0"' \
      '    }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace

# Dependencies first, so editing mavrl/ does not reinstall torch.
COPY pyproject.toml setup.py README.md ./
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir \
        "gymnasium>=0.29.0" "mujoco>=3.0.0" "numpy>=1.21.0" \
        "stable-baselines3>=2.0.0" "matplotlib>=3.5.0" \
        "imageio>=2.31.0" "imageio-ffmpeg>=0.4.9" "tqdm>=4.65.0"

COPY . .
RUN python3 -m pip install --no-cache-dir --no-deps -e .

# The first thing to run in a fresh container, and the last thing to blame.
CMD ["python3", "-m", "mavrl.glcheck"]
