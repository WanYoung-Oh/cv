"""
PyTorch Lightning Module for document image classification
"""

from typing import Optional

import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class DocumentClassifierModule(pl.LightningModule):
    """문서 이미지 분류 LightningModule"""

    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 17,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        warmup_epochs: int = 5,
        epochs: int = 50,
        warmup_start_factor: float = 0.01,
        scheduler_eta_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.warmup_start_factor = warmup_start_factor
        self.scheduler_eta_min = scheduler_eta_min

        # 모델 로드 (timm 1.0+ 호환)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # class_weights를 buffer로 등록 (Lightning이 디바이스 자동 관리)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights
        )

        # torchmetrics (디바이스 자동 동기화, 메모리 효율적)
        metrics_kwargs = {"num_classes": num_classes, "average": "macro"}
        self.train_acc = MulticlassAccuracy(**metrics_kwargs)
        self.train_f1 = MulticlassF1Score(**metrics_kwargs)

        self.val_acc = MulticlassAccuracy(**metrics_kwargs)
        self.val_f1 = MulticlassF1Score(**metrics_kwargs)
        self.val_precision = MulticlassPrecision(**metrics_kwargs)
        self.val_recall = MulticlassRecall(**metrics_kwargs)

        self.test_acc = MulticlassAccuracy(**metrics_kwargs)
        self.test_f1 = MulticlassF1Score(**metrics_kwargs)
        self.test_precision = MulticlassPrecision(**metrics_kwargs)
        self.test_recall = MulticlassRecall(**metrics_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # .clone()은 새로운 메모리 공간을 할당하여 stride 문제를 원천 차단합니다.
        return self.model(x).clone().contiguous()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 1. 로짓과 타겟의 메모리 상태를 강제로 정렬
        logits = logits.float().contiguous()
        loss = self.loss_fn(logits, y)
    
        # 2. 역전파가 발생할 loss 텐서도 한 번 더 정렬 (매우 드물지만 필요한 경우 있음)
        loss = loss.contiguous()

        # 3. 평가지표 계산 시 .detach()를 사용하여 미분 그래프와 완전히 분리
        with torch.no_grad():
            preds = logits.detach().argmax(dim=1)
            self.train_acc(preds, y)
            self.train_f1(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_precision", self.test_precision)
        self.log("test_recall", self.test_recall)

    def configure_optimizers(self):
        """옵티마이저 및 스케줄러 설정"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
     
        # 1. Warmup: lr이 점진적으로 증가
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.warmup_start_factor,
            total_iters=self.warmup_epochs
        )

        # 2. Main: Warmup 이후 감쇄
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=self.scheduler_eta_min
        )

        # 3. 결합
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        return logits.argmax(dim=1)
