# Keep the same style as comical: NVIDIA pytorch base
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Same “clean pip” environment pattern
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# OS deps (minimal, but enough for common wheels + nibabel/image stuff)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade build tooling (very similar to typical comical setup)
RUN python -m pip install --upgrade pip setuptools wheel

# Critical: avoid numpy 2.x issues
RUN python -m pip install "numpy<2"

# Install python requirements (SwinBrain has requirements.txt at repo root)
COPY requirements.txt /opt/SwinBrain/requirements.txt
RUN python -m pip install -r /opt/SwinBrain/requirements.txt

# Re-enforce numpy<2 in case requirements pull numpy>=2
RUN python -m pip install --upgrade --force-reinstall "numpy<2"

# Default workdir (nice for later)
WORKDIR /work

# A simple default command
CMD ["python", "-c", "import numpy as np; import torch; print('NumPy', np.__version__); print('Torch', torch.__version__)"]
