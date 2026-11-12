# CV Training Service

Minimal PyTorch service for supervised computer-vision training. The training loop is shared. Data sources, annotation formats, preprocessing, and artifact storage are injected from a YAML config.

CPU is the default device. Set `device` to `cuda` or `auto` to train on a GPU. Mixed precision runs only when the resolved device is CUDA.

Evaluation and inference are intentionally not part of this service.

## Layout

```text
src/data_loading/     sources, annotation readers, PyTorch dataset
src/processing/       preprocessing, augmentation, train/validation split
src/training/         BaseTrainer, concrete trainers, reference models
src/exporting/        local and S3 artifact stores
src/cli.py            entry point
configs/              example training configs
requirements.txt      runtime dependencies
Dockerfile            PyTorch 2.15 CUDA 13.2 runtime
```

## Install

```bash
pip install -r requirements.txt
pip install pytest
```

Run commands from the repository root with `src` on `PYTHONPATH`. The Docker image sets that for you.

## Run

Classification expects this layout under `data/classification/`:

```text
images/<id>.jpg
annotations/<id>.json    {"label": 1}
```

Detection expects the same layout with boxes:

```json
{"boxes": [[10, 12, 40, 48]], "labels": [0]}
```

Boxes are `x1, y1, x2, y2` in the original image's pixel coordinates. Labels are zero-based foreground classes.

```bash
set PYTHONPATH=src
python -m cli --config configs/classification.yaml
python -m cli --config configs/object_detection.yaml
```

The image is the official PyTorch 2.15 runtime with CUDA 13.2. It trains on CPU, and on a GPU when the container is started with the NVIDIA runtime. `pip install -r requirements.txt` keeps the image's CUDA build of PyTorch when that build already satisfies `torch>=2.2`.

```bash
docker build -t cv-training-service .
docker run --rm -v ${PWD}:/workspace cv-training-service --config /workspace/configs/classification.yaml
```

Artifacts written for a run:

- `config.yaml` — the config used for the run
- `metadata.json` — device, sample ids, split sizes
- `metrics.jsonl` — one record per epoch (`epoch`, `train_loss`, `duration_seconds`, `num_batches`)
- `checkpoint.pt` — model, optimizer, and last metrics, used when `training.resume` is true
- `weights.pt` — model weights on CPU

To store artifacts in S3, point `persistence.store` at `exporting.s3.S3ArtifactStore` with `bucket` and `prefix`. To read training data from S3, point `data.source` at `data_loading.s3.S3DataSource`. The object layout matches the local `images/` and `annotations/` directories.

## Add a trainer

Subclass `BaseTrainer`, implement `compute_loss`, and set `trainer` in the config to that class path. Set `batch_collate` on the subclass when targets cannot be stacked by the default collate. No change to the training loop, data sources, or stores is required.

`TinyDetector` and `GridDetectionLoss` are reference implementations: each box is assigned to the grid cell that contains its center. Replace the model class path with a different `nn.Module` that returns `{"logits": (B, H, W, C), "boxes": (B, H, W, 4)}` if you want another head behind the same trainer.

GPU transforms are not implemented. `preparation.transform_device` must stay `cpu` until a pipeline with the same `(image, target)` signature is added in `processing`.
