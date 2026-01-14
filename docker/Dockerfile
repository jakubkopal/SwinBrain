# GPU-ready base (CUDA 11.8 + PyTorch)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps (keep minimal; add only if you actually need compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency file(s) first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install deps
RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
 && python -m pip install --no-cache-dir "numpy<2" \
 && python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python -m pip install --no-cache-dir --force-reinstall "numpy<2"

# Now copy the actual code (keeps rebuilds smaller when only code changes)
COPY . /app

# Quick self-tests
RUN python -c "import sys; print(sys.version)" \
 && python -c "import numpy as np; print('numpy', np.__version__)" \
 && python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

CMD ["python", "-c", "import numpy as np, torch; print('NumPy', np.__version__); print('Torch', torch.__version__)"]
