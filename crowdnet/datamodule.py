from typing import Any
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import JHUCrowdDataset, get_train_transform, test_transform
from .datautils import seed_worker


class CrowdDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_root: Path,
        batch_size: int = 8,
        num_workers: int = 8,
        input_size: int = 512,
        min_crowd_size: int = 50,
        density_scale_factor: int = 8,
    ) -> None:
        super().__init__()

        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.min_crowd_size = min_crowd_size
        self.density_scale_factor = density_scale_factor

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        batch[0] = batch[0].to(device)
        batch[1] = batch[1].to(device)
        return batch

    def train_dataloader(self) -> DataLoader:
        dataset = JHUCrowdDataset(
            dataset_root=self.dataset_root,
            subset_name="train",
            min_size=512,
            max_size=1536,
            transform=get_train_transform(input_w=self.input_size, input_h=self.input_size),
            min_crowd_size=self.min_crowd_size,
            scale_factor=self.density_scale_factor
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=JHUCrowdDataset.collate_fn,
            drop_last=True,
            shuffle=True,
            worker_init_fn=seed_worker
        )

    def val_dataloader(self) -> DataLoader:
        dataset = JHUCrowdDataset(
            dataset_root=self.dataset_root,
            subset_name="val",
            min_size=512,
            max_size=1536,
            transform=test_transform,
            min_crowd_size=self.min_crowd_size,
            scale_factor=self.density_scale_factor
        )

        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=JHUCrowdDataset.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        dataset = JHUCrowdDataset(
            dataset_root=self.dataset_root,
            subset_name="test",
            min_size=512,
            max_size=1536,
            transform=test_transform,
            min_crowd_size=self.min_crowd_size,
            scale_factor=self.density_scale_factor
        )

        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=JHUCrowdDataset.collate_fn
        )
