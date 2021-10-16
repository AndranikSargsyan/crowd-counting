import argparse
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanAbsoluteError

from crowdnet.datamodule import CrowdDataModule
from crowdnet.model import CSRNet


class LitCrowdNet(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()

        self.model = CSRNet()

        self.valid_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def loss(self, pred_dens_map: Tensor, gt_dens_map: Tensor) -> Tensor:
        return F.mse_loss(pred_dens_map, gt_dens_map, reduction="sum") / (torch.sum(gt_dens_map) + 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.model.parameters(), 5e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": StepLR(optimizer, step_size=2, gamma=0.5)
            },
        }

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        output = self.forward(batch[0])
        loss = self.loss(output, batch[1])
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        output = self.forward(batch[0])
        loss = self.loss(output, batch[1])

        gt_counts = torch.sum(batch[1].detach().squeeze(1), dim=[1, 2])
        pred_counts = torch.sum(torch.clip(output.detach().squeeze(1), min=0), dim=[1, 2])

        self.valid_mae(pred_counts, gt_counts)

        self.log("valid_loss", loss.item(), on_step=False, on_epoch=True)
        self.log("valid_mae", self.valid_mae, on_step=False, on_epoch=True)

    def test_step(self, batch: List[Tensor], batch_idx: int) -> None:
        output = self.forward(batch[0])
        loss = self.loss(output, batch[1])

        gt_counts = torch.sum(batch[1].detach().squeeze(1), dim=[1, 2])
        pred_counts = torch.sum(torch.clip(output.detach().squeeze(1), min=0), dim=[1, 2])

        self.test_mae(pred_counts, gt_counts)

        self.log("test_loss", loss.item(), on_step=False, on_epoch=True)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, help="Path to JHU Crowd++ dataset.", default="./jhu_crowd_v2.0")
    parser.add_argument("--checkpoints-dir", type=Path, help="Dir to save checkpoints in.", default="./checkpoints")
    parser.add_argument("--max-epochs", type=int, help="Maximum number of epochs.", default=50)
    return parser.parse_args()


def main():
    args = get_args()

    seed_everything(42)
    dm = CrowdDataModule(args.dataset_root)
    lit_model = LitCrowdNet()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename="crowdnet-{epoch:02d}-{valid_mae:.2f}",
        monitor="valid_mae",
        mode="min",
        verbose=True,
        save_top_k=5
    )

    trainer = pl.Trainer(
        gpus=[0],
        fast_dev_run=False,
        resume_from_checkpoint=None,
        max_epochs=args.max_epochs,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        deterministic=True
    )

    trainer.fit(lit_model, datamodule=dm)
    trainer.test(lit_model, datamodule=dm)
    torch.save(lit_model.model, args.save_path)


if __name__ == "__main__":
    main()
