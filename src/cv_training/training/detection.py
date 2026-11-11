"""Reference object-detection trainer.

Each box is assigned to the grid cell that contains its center. The loss is
background-aware classification plus L1 on the positive cells. This is a
minimal supervised objective so a detection run can execute end to end. It is
not a production detector and it does not score detections.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from cv_training.training.base import BaseTrainer


class GridDetectionLoss(torch.nn.Module):
    def __init__(self, image_size: tuple[int, int], box_weight: float = 1.0) -> None:
        super().__init__()
        self.image_height = int(image_size[0])
        self.image_width = int(image_size[1])
        self.box_weight = float(box_weight)

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        logits = outputs["logits"]
        pred_boxes = outputs["boxes"]
        batch, grid_h, grid_w, _classes = logits.shape
        cls_target = torch.zeros(batch, grid_h, grid_w, dtype=torch.long, device=logits.device)
        box_target = torch.zeros(batch, grid_h, grid_w, 4, device=logits.device)
        positive = torch.zeros(batch, grid_h, grid_w, dtype=torch.bool, device=logits.device)
        self._assign_targets(targets, cls_target, box_target, positive, grid_h, grid_w)

        class_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), cls_target.reshape(-1))
        if bool(positive.any().item()):
            box_loss = F.l1_loss(pred_boxes[positive], box_target[positive])
        else:
            box_loss = pred_boxes.sum() * 0.0
        return class_loss + self.box_weight * box_loss

    def _assign_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        cls_target: torch.Tensor,
        box_target: torch.Tensor,
        positive: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> None:
        height = float(self.image_height)
        width = float(self.image_width)
        for index, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]
            if boxes.numel() == 0:
                continue
            x1, y1, x2, y2 = boxes.unbind(dim=-1)
            cell_x = torch.clamp(((x1 + x2) / 2 / width * grid_w).long(), 0, grid_w - 1)
            cell_y = torch.clamp(((y1 + y2) / 2 / height * grid_h).long(), 0, grid_h - 1)
            normalized = torch.stack((x1 / width, y1 / height, x2 / width, y2 / height), dim=-1).clamp(0, 1)
            cls_target[index, cell_y, cell_x] = labels.to(dtype=torch.long) + 1
            box_target[index, cell_y, cell_x] = normalized
            positive[index, cell_y, cell_x] = True


class ObjectDetectionTrainer(BaseTrainer):
    """Train a dense detector that returns ``{"logits", "boxes"}``."""

    @staticmethod
    def batch_collate(
        batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        images = torch.stack([image for image, _target in batch], dim=0)
        targets = [target for _image, target in batch]
        return images, targets

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        loss_cfg = self.config.loss
        if loss_cfg is not None and loss_cfg.name != "grid_detection":
            raise ValueError("ObjectDetectionTrainer expects loss.name to be 'grid_detection'.")
        params = {} if loss_cfg is None else dict(loss_cfg.params)
        unknown = set(params) - {"box_weight"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported detection loss params: {names}.")
        self.criterion = GridDetectionLoss(
            image_size=self.config.preparation.image_size,
            box_weight=float(params.get("box_weight", 1.0)),
        ).to(self.device)

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        return self.criterion(outputs, targets)
