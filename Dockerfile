FROM pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/cv-training-service/src

WORKDIR /opt/cv-training-service

COPY requirements.txt .
# The base image already provides CUDA torch 2.15, which satisfies torch>=2.2.
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "cli"]
