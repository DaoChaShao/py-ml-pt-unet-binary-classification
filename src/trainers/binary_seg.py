#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/17 23:06
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   binary_seg.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from torch import nn, no_grad, device, long, sigmoid, Tensor, save

from src.utils.PT import get_device, DataLoader, TorchDataLoader

WIDTH: int = 64


class BinaryIoU(nn.Module):
    """
    Compute binary IoU for both background & foreground.
    pred: logits or probs — will auto sigmoid
    label: 0/1 mask
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self._threshold = threshold
        self._eps = eps

    def forward(self, pred, target):
        pred = sigmoid(pred)
        pred_bin = (pred > self._threshold).float()

        # Foreground IoU
        inter_fg = (pred_bin * target).sum()
        union_fg = ((pred_bin + target) > 0).sum()
        iou_fg = (inter_fg + self._eps) / (union_fg + self._eps)

        # Background IoU
        pred_bg = 1 - pred_bin
        target_bg = 1 - target

        inter_bg = (pred_bg * target_bg).sum()
        union_bg = ((pred_bg + target_bg) > 0).sum()
        iou_bg = (inter_bg + self._eps) / (union_bg + self._eps)

        return iou_bg.item(), iou_fg.item()


class Evaluator:
    """
    Compute precision, recall, F1 by confusion matrix.
    Works for binary masks.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # ensure integers
        self.TP: int = 0
        self.FP: int = 0
        self.FN: int = 0
        self.TN: int = 0

    def update(self, pred, target):
        """
        pred and target are expected to be tensors or numpy arrays of 0/1,
        same shape. We'll flatten and accumulate counts as Python ints.
        """
        # convert to tensor-like comparisons and sum to int
        # pred and target may be torch tensors or numpy arrays
        try:
            # assume torch tensors first
            tp = int(((pred == 1) & (target == 1)).sum().item())
            tn = int(((pred == 0) & (target == 0)).sum().item())
            fp = int(((pred == 1) & (target == 0)).sum().item())
            fn = int(((pred == 0) & (target == 1)).sum().item())
        except Exception:
            # fallback for numpy arrays
            import numpy as _np
            pred_a = _np.asarray(pred).astype(_np.int32).flatten()
            target_a = _np.asarray(target).astype(_np.int32).flatten()
            tp = int(((pred_a == 1) & (target_a == 1)).sum())
            tn = int(((pred_a == 0) & (target_a == 0)).sum())
            fp = int(((pred_a == 1) & (target_a == 0)).sum())
            fn = int(((pred_a == 0) & (target_a == 1)).sum())

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute(self):
        # return floats and confusion counts as ints
        precision = float(self.TP / (self.TP + self.FP + 1e-8)) if (self.TP + self.FP) > 0 else 0.0
        recall = float(self.TP / (self.TP + self.FN + 1e-8)) if (self.TP + self.FN) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall + 1e-8)) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": int(self.TP),
            "false_positive": int(self.FP),
            "false_negative": int(self.FN),
            "true_negative": int(self.TN),
            "confusion_matrix": {
                "TP": int(self.TP), "FP": int(self.FP),
                "FN": int(self.FN), "TN": int(self.TN)
            }
        }

    def print_metrics(self):
        metrics = self.compute()

        print("*" * WIDTH)
        print("Segmentation Evaluation Metrics")
        print("-" * WIDTH)
        print(f"TP: {metrics['true_positive']}, FP:  {metrics['false_positive']}")
        print(f"FN: {metrics['false_negative']}, TN: {metrics['true_negative']}\n")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print("*" * WIDTH)
        print()


class UNetSegmentationTrainer(QObject):
    """ Trainer class for managing training process """
    losses: Signal = Signal(int, float, float, float, float)

    def __init__(self,
                 model: nn.Module,
                 optimiser,
                 criterion,
                 scheduler=None,
                 accelerator: str = "auto"
                 ) -> None:
        super().__init__()
        self._accelerator = get_device(accelerator)
        self._model = model.to(device(self._accelerator))
        self._optimiser = optimiser
        self._criterion = criterion
        self._scheduler = scheduler
        self._evaluator = Evaluator()

    def _epoch_train(self, dataloader: DataLoader | TorchDataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _batch_num: float = 0.0
        for features, labels in dataloader:
            features = features.to(device(self._accelerator))
            labels = labels.to(device(self._accelerator))

            # Ensure the shape dimension of labels is right [B, 1, H, W]
            if len(labels.shape) == 3:  # [B, H, W]
                labels = labels.unsqueeze(1)  # -> [B, 1, H, W]

            self._optimiser.zero_grad()
            outputs = self._model(features)  # [B, 1, H, W]
            # print(outputs.shape, masks.shape)

            loss = self._criterion(outputs, labels)
            loss.backward()
            # Clip gradients
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimiser.step()

            _loss += loss.item()
            _batch_num += 1.0

        return _loss / _batch_num

    def _epoch_valid(self, dataloader):
        # Set model to evaluation mode
        self._model.eval()
        self._evaluator.reset()

        _loss: float = 0.0
        _batch_num: float = 0.0
        _pixel_acc: float = 0
        _pixel_total: float = 0

        _total_intersection_fg: float = 0
        _total_union_fg: float = 0
        _total_intersection_bg: float = 0
        _total_union_bg: float = 0

        with no_grad():
            for features, labels in dataloader:
                features = features.to(device(self._accelerator))
                labels = labels.to(device(self._accelerator))

                # Ensure the shape dimension of labels is right [B, 1, H, W]
                if len(labels.shape) == 3:
                    labels = labels.unsqueeze(1)

                outputs = self._model(features)  # [B, 1, H, W]
                loss = self._criterion(outputs, labels)

                # Binary classification Prediction：sigmoid + threshold
                predictions = (sigmoid(outputs) > 0.5).float()

                # Alternatively, flatten the tensors for easier computation
                pred_flat = predictions.squeeze(1)  # [B, H, W]
                labels_flat = labels.squeeze(1)  # [B, H, W]

                _loss += loss.item()

                # Calculate pixel accuracy
                _pixel_acc += (predictions == labels).sum().item()
                _pixel_total += labels.numel()

                # Calculate the IoU of background class (fixed union via OR)
                pred_bg = 1 - pred_flat
                target_bg = 1 - labels_flat

                intersection_bg = (pred_bg * target_bg).sum().item()
                union_bg = ((pred_bg + target_bg) > 0).sum().item()
                _total_intersection_bg += intersection_bg
                _total_union_bg += union_bg

                intersection_fg = (pred_flat * labels_flat).sum().item()
                union_fg = ((pred_flat + labels_flat) > 0).sum().item()
                _total_intersection_fg += intersection_fg
                _total_union_fg += union_fg

                _batch_num += 1.0

                # Update evaluator with binary masks (flattened ints)
                self._evaluator.update(predictions.cpu(), labels.cpu())

        # Print evaluator metrics
        self._evaluator.print_metrics()

        # Calculate the mean IoU and pixel accuracy
        iou_fg = _total_intersection_fg / (_total_union_fg + 1e-8)
        iou_bg = _total_intersection_bg / (_total_union_bg + 1e-8)
        mean_iou = (iou_fg + iou_bg) / 2

        print("*" * WIDTH)
        print("Binary Segmentation Metrics")
        print("-" * WIDTH)
        print(f"- Background IoU:               {iou_bg * 100:.4f}%")
        print(f"- Foreground IoU:               {iou_fg * 100:.4f}%")
        print(f"- mean Intersection over Union: {mean_iou * 100:.4f}%")
        print("*" * WIDTH)
        print()

        return _loss / _batch_num, mean_iou, _pixel_acc / _pixel_total

    @staticmethod
    def _iou_per_class(predictions: Tensor, labels: Tensor, num_classes: int):
        """
        返回两个 Tensor，shape = (num_classes,)
        - intersections[c] = intersection pixels for class c
        - unions[c] = union pixels for class c
        """
        intersections = predictions.new_zeros(num_classes, dtype=long)
        unions = predictions.new_zeros(num_classes, dtype=long)

        for c in range(num_classes):
            pred_mask = (predictions == c)
            true_mask = (labels == c)

            inter = (pred_mask & true_mask).sum().item()
            uni = (pred_mask | true_mask).sum().item()

            intersections[c] = inter
            unions[c] = uni

        return intersections, unions

    def fit(self,
            train_loader: DataLoader | TorchDataLoader, valid_loader: DataLoader | TorchDataLoader,
            epochs: int, model_save_path: str | None = None
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :return: None
        """
        _best_mIoU: float = 0.0
        _patience: int = 5
        _patience_counter: int = 0

        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss, mIoU, pixel_acc = self._epoch_valid(valid_loader)

            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss, mIoU, pixel_acc)

            print(
                f"Epoch [{epoch + 1}/{epochs}]:\n"
                f"- Train Loss:                    {train_loss:.4f}\n"
                f"- Valid Loss:                    {valid_loss:.4f}\n"
                f"- Pixel Accuracy:                {pixel_acc * 100:.4f}%"
            )

            # Save the model if it has the best validation loss so far
            if mIoU > _best_mIoU + 1e-4:  # Save it even with tiny improvement
                _best_mIoU = mIoU
                _patience_counter = 0
                if model_save_path is not None:
                    save(self._model.state_dict(), model_save_path)
                    print(f"√ Model's parameters saved to {model_save_path}\n")
            else:
                _patience_counter += 1
                print(f"× No improvement [{_patience_counter}/{_patience}]\n")
                if _patience_counter >= _patience:
                    print("*" * WIDTH)
                    print("Early Stopping Triggered")
                    print("-" * WIDTH)
                    print(f"Early stopping at epoch {epoch}, best mIoU: {_best_mIoU * 100:.2f}%")
                    print("*" * WIDTH)
                    print()
                    break

            if self._scheduler is not None:
                self._scheduler.step(mIoU)


if __name__ == "__main__":
    pass
