FROM pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/cv-training-service/src \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# The base image Python is externally managed and already contains CUDA torch,
# numpy, pillow, and PyYAML. A venv with system site-packages reuses those
# builds; pip then installs only requirements that are not already present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv --system-site-packages

WORKDIR /opt/cv-training-service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "main"]
