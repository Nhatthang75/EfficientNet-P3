# Use a lightweight python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Optimize PyTorch & system memory usage for 512MB RAM limits
ENV MALLOC_ARENA_MAX=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PYTORCH_MALLOC_CONF=max_split_size_mb:32
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Pre-download the model weights during Docker build to save RAM and avoid network/download overhead at runtime.
# Requires HF_TOKEN as a build argument: docker build --build-arg HF_TOKEN=your_token ...
ARG HF_TOKEN
RUN python -c "import os; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='chrisnguyenx/EfficientNet-P3', filename='efficientnet_b4_cbam_fold1.pth', local_dir='.', token='${HF_TOKEN}')"

# Copy code
COPY preprocessing.py .
COPY main_api.py .

# Expose port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
