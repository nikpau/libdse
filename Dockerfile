# ── Base image ────────────────────────────────────────────────────────────────
# python:3.12-slim keeps the image small while matching the project requirement.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libsndfile1    — required by librosa / soundfile to read audio files
# ffmpeg         — fallback audio decoder used by librosa
# libgomp1       — OpenMP runtime required by PyTorch CPU kernels
# build-essential — gcc/g++ needed to compile pesq (C extension); removed after
#                   pip install to keep the final image slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the package manifest first so the dependency layer is cached
# independently from source-code changes.
COPY pyproject.toml .

# Install CPU-only PyTorch first (avoids pulling the much larger CUDA build
# that the default PyPI wheel would select).
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# ── Application source ────────────────────────────────────────────────────────
COPY src/ src/
COPY models/ models/

# Install the package and all dependencies declared in pyproject.toml.
RUN pip install --no-cache-dir . \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860

# Gradio serves on 0.0.0.0:7860 (set explicitly in the showcase script).
CMD ["python", "-m", "aese.showcases.simpleAE_logmag_nc"]
