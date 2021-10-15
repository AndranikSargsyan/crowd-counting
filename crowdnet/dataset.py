from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from crowdnet.datautils import (
    load_gt, load_image, load_dataset_paths, preprocess_transform, random_crop, resize_img, generate_density_map
)


def get_train_transform(
    input_h: int = 512,
    input_w: int = 512
) -> Callable[[Image.Image, np.ndarray], Tuple[Tensor, np.ndarray]]:
    def transform(
        img: Image.Image,
        density_map: np.ndarray,
    ) -> Tuple[Tensor, np.ndarray]:
        cropped_img, cropped_density_map = random_crop(img, density_map, input_h=input_h, input_w=input_w)

        aug_result = albu.Compose([
            albu.RandomBrightnessContrast(p=0.4),
            albu.ISONoise(p=0.25),
            albu.HorizontalFlip(p=0.5)
        ])(image=np.array(cropped_img), mask=cropped_density_map)

        cropped_img = Image.fromarray(aug_result["image"])
        cropped_density_map = aug_result["mask"]

        cropped_img = preprocess_transform(cropped_img)

        return cropped_img, cropped_density_map

    return transform


def test_transform(img: Image.Image, density_map: np.ndarray) -> Tuple[Tensor, np.ndarray]:
    img = preprocess_transform(img)
    return img, density_map


class JHUCrowdDataset(Dataset):
    img_extensions = {".jpg"}

    def __init__(
        self,
        dataset_root: Path,
        subset_name: str = "train",
        transform: Optional[Callable[[Image.Image, np.ndarray], Tuple[Tensor, np.ndarray]]] = None,
        scale_factor: int = 1,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        min_crowd_size: int = 0,
        force_pregenerate: bool = False,
        num_workers: int = 8,
        cache_root: Path = Path("./cache"),
        cache: bool = True
    ) -> None:
        super(JHUCrowdDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset_name = subset_name
        self.transform = transform
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.max_size = max_size
        self.min_crowd_size = min_crowd_size
        self.num_workers = num_workers
        self.cache_dir = cache_root / subset_name
        self.cache = cache

        img_labels_df = self.load_img_labels(dataset_root / subset_name)
        self.img_labels_df = img_labels_df[img_labels_df["count"] >= min_crowd_size]

        if force_pregenerate and cache:
            self.pregenerate_density_maps()

    def load_img_labels(self, dataset_root: Path) -> pd.DataFrame:
        # Load image_labels.txt
        img_info_df = pd.read_csv(
            dataset_root / "image_labels.txt",
            names=["name", "count", "scene_type", "weather_condition", "distractor"],
            dtype={"name": str, "count": int, "scene_type": str, "weather_condition": int, "distractor": int}
        )
        img_info_df.set_index("name", inplace=True)

        # Load image & gt path combinations"
        img_gt_paths = load_dataset_paths(dataset_root, self.img_extensions)
        names = [img_path.stem for img_path, _ in img_gt_paths]

        img_gt_paths_df = pd.DataFrame(img_gt_paths, columns=["img_path", "gt_path"], index=names)
        gt_labels = pd.Series([
            load_gt(gt_path) for gt_path in img_gt_paths_df["gt_path"]], name="gt_labels", index=names)
        gt_counts = pd.Series([len(gt) for gt in gt_labels], index=names)

        # Check integrity
        assert len(img_info_df) == len(img_gt_paths)
        assert len(img_gt_paths_df.index.difference(img_info_df.index)) == 0
        assert all(img_info_df["count"].eq(gt_counts))

        return pd.concat([img_gt_paths_df, gt_labels, img_info_df], axis=1)

    def pregenerate_density_maps(self) -> None:
        with Pool(self.num_workers) as p:
            p.map(self.__getitem__, range(len(self)))

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.ndarray]:
        sample = self.img_labels_df.iloc[index]

        cached_img_path = self.cache_dir / f"{sample.name}.jpg"
        cached_density_path = self.cache_dir / f"{sample.name}.npy"
        if self.cache and cached_density_path.exists() and cached_img_path.exists():
            img = load_image(cached_img_path)
            density_map = np.load(str(cached_density_path))
        else:
            img = load_image(sample["img_path"])
            keypoints = np.empty((0, 2))
            if len(sample["gt_labels"] > 0):
                keypoints = sample["gt_labels"][:, :2]

            if self.min_size is not None and self.max_size is not None:
                img, resize_ratio = resize_img(img, min_size=self.min_size, max_size=self.max_size)
                keypoints = (keypoints * resize_ratio).astype(np.int32)
            density_map = generate_density_map(img.size, keypoints)

            if self.cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(self.cache_dir / f"{sample.name}.npy"), density_map)
                img.save(str(self.cache_dir / f"{sample.name}.jpg"))

        if self.transform is not None:
            img, density_map = self.transform(img, density_map)

        density_map = cv2.resize(
            density_map,
            (density_map.shape[1] // self.scale_factor, density_map.shape[0] // self.scale_factor),
            interpolation=cv2.INTER_CUBIC
        ) * (self.scale_factor ** 2)

        return img, density_map

    def __len__(self) -> int:
        return len(self.img_labels_df)

    @staticmethod
    def collate_fn(samples: List[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        img_batch = torch.stack([sample[0] for sample in samples], 0)
        density_map_batch = torch.stack([torch.from_numpy(sample[1]).unsqueeze(0) for sample in samples], 0)
        return [img_batch, density_map_batch]
