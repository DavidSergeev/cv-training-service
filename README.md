# CV Training Service

Minimal PyTorch service for supervised computer-vision training. The training loop is shared. Data sources, annotation formats, preprocessing, and artifact storage are injected from a YAML config.

CPU is the default device. Set `device` to `cuda` or `auto` to train on a GPU. Mixed precision runs only when the resolved device is CUDA.

Evaluation and inference are intentionally not part of this service.

## Layout

```text
src/cv_training/
  builder.py              wires a config into a trainer
  training/base.py        BaseTrainer lifecycle
  training/supervised.py  generic injected-loss trainer
  training/detection.py   reference object-detection trainer
  data/                   source interface, local and S3 loaders, annotations
  preparation/            resize, normalize, flip, train/validation split
  persistence/            artifact-store interface, local and S3 stores
  models/                 small reference models
configs/                  example training configs
```

## Install

```bash
pip install -e ".[dev]"
```

S3 backends need the optional extra:

```bash
pip install -e ".[s3]"
```

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
python -m cv_training --config configs/classification.yaml
python -m cv_training --config configs/object_detection.yaml
```

Artifacts written for a run:

- `config.yaml` — the config used for the run
- `metadata.json` — device, sample ids, split sizes
- `metrics.jsonl` — one record per epoch (`epoch`, `train_loss`, `duration_seconds`, `num_batches`)
- `checkpoint.pt` — model, optimizer, and last metrics, used when `training.resume` is true
- `weights.pt` — model weights on CPU

To store artifacts in S3, point `persistence.store` at `cv_training.persistence.s3.S3ArtifactStore` with `bucket` and `prefix`. To read training data from S3, point `data.source` at `cv_training.data.s3.S3DataSource`. The object layout matches the local `images/` and `annotations/` directories.

## Add a trainer

Subclass `BaseTrainer`, implement `compute_loss`, and set `trainer` in the config to that class path. Set `batch_collate` on the subclass when targets cannot be stacked by the default collate. No change to the training loop, data sources, or stores is required.

`TinyDetector` and `GridDetectionLoss` are reference implementations: each box is assigned to the grid cell that contains its center. Replace the model class path with a different `nn.Module` that returns `{"logits": (B, H, W, C), "boxes": (B, H, W, 4)}` if you want another head behind the same trainer.

GPU transforms are not implemented. `preparation.transform_device` must stay `cpu` until a pipeline with the same `(image, target)` signature is added.
