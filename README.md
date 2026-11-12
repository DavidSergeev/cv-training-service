# CV Training Service

Minimal PyTorch service for supervised computer-vision training. The training loop is shared. Data sources, annotation formats, preprocessing, and artifact storage are injected from a YAML config.

CPU is the default device. Set `device` to `cuda` or `auto` to train on a GPU. Mixed precision runs only when the resolved device is CUDA.

Evaluation and inference are intentionally not part of this service.

## Layout

```text
src/data_loading/     sources, annotation readers, PyTorch dataset
src/model_loading/    PyTorch checkpoints and Hugging Face models
src/processing/       preprocessing, augmentation, train/validation split
src/training/         BaseTrainer, concrete trainers, reference models
src/exporting/        local and S3 artifact stores
src/main.py           entry point
configs/service.yaml  database URL for id: jobs
configs/              example training documents
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

`configs/classification.yaml` fine-tunes `google/mobilenet_v2_1.0_224` on a capped stream of the Hugging Face `AI-Lab-Makerere/beans` dataset. Another image classifier uses the same model block with its own `repo_id`, `num_labels`, and `preparation.image_size`. A `data.source` of `kind: huggingface` reads `image_column` and an integer `label_column`, and `max_samples` stops the stream. SoccerNet repositories use those same fields only when a revision actually exposes those columns. Broadcast video and frame archives do not.

A `kind: local` classification dataset expects this layout under `data/classification/`:

```text
images/<id>.jpg
annotations/<id>.json    {"label": 1}
```

Detection expects the same layout with boxes:

```json
{"boxes": [[10, 12, 40, 48]], "labels": [0]}
```

Boxes are `x1, y1, x2, y2` in the original image's pixel coordinates. Labels are zero-based foreground classes.

The container reads `TRAIN_CONFIG` from the environment. The value is either `path:<yaml file>` or `id:<uuid>`. An unset or empty value exits with status 1. There is no default.

`path:` reads a YAML training document. `id:` loads the same document from `train_configs.config`, using `database_url` in `configs/service.yaml`. The row id must equal the `id` field inside the document.

```sql
CREATE TABLE train_configs (
    id uuid PRIMARY KEY,
    config jsonb NOT NULL
);
```

```bash
set PYTHONPATH=src
set TRAIN_CONFIG=path:configs/classification.yaml
python -m main
```

The image is the official PyTorch 2.15 runtime with CUDA 13.2. It trains on CPU, and on a GPU when the container is started with the NVIDIA runtime. Dependencies install into a virtual environment that reuses the image's CUDA PyTorch, NumPy, Pillow, and PyYAML, so pip only adds packages the image does not already provide.

```bash
docker build -t cv-training-service .
docker run --rm -e TRAIN_CONFIG=path:/workspace/configs/classification.yaml -v ${PWD}:/workspace cv-training-service
```

Artifacts written for a run:

- `config.yaml` — the config used for the run
- `metadata.json` — device, sample ids, split sizes
- `metrics.jsonl` — one record per epoch (`epoch`, `train_loss`, `duration_seconds`, `num_batches`)
- `checkpoint.pt` — model, optimizer, and last metrics, used when `training.resume` is true
- `weights.pt` — model weights on CPU

`model.format` is `torch_checkpoint` or `huggingface`. `mode: finetune` loads the checkpoint or Hub weights. `mode: from_scratch` with a `state_dict` builds `torch.architecture` and does not read the checkpoint file. `trainable: head_only` freezes every parameter whose name is not a head (`head`, `fc`, `classifier`, `classification`, `boxes`).

To store artifacts in S3, set `persistence.kind` to `s3` with `bucket` and `prefix`. To read training data from S3, set `data.source.kind` to `s3`. The object layout matches the local `images/` and `annotations/` directories.

## Add a trainer

`task: classification` uses `SupervisedTrainer`. `task: detection` uses `ObjectDetectionTrainer`. Subclass `BaseTrainer` and implement `compute_loss` when a task needs a different objective. Set `batch_collate` on the subclass when targets cannot be stacked by the default collate.

`TinyDetector` and `GridDetectionLoss` are reference implementations: each box is assigned to the grid cell that contains its center. A `state_dict` checkpoint names that class, or another `nn.Module` with the same `{"logits": (B, H, W, C), "boxes": (B, H, W, 4)}` output, in `torch.architecture.class_path`.

Preprocessing runs on CPU. Hugging Face classification models return an object with `logits`, and `SupervisedTrainer` uses that tensor as the loss input.
