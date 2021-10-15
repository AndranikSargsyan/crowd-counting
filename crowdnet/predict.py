import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

from crowdnet.dataset import JHUCrowdDataset
from crowdnet.datautils import load_image, preprocess_transform, resize_img, count_from_density_map
from crowdnet.model import CSRNet
from crowdnet.train import LitCrowdNet


def predict_density_map(img: Image.Image, model: nn.Module) -> np.ndarray:
    img, resize_ratio = resize_img(img)
    img_input = preprocess_transform(img).unsqueeze(dim=0)
    density_map = model(img_input).detach().cpu().numpy()[0][0]
    return density_map


def get_args():
    parser = argparse.ArgumentParser()

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--ckpt-path", type=str, default="", help="Checkpoint path.")
    model_group.add_argument("--model-path", type=str, default="", help="Saved model path.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on.")
    parser.add_argument("--img-path", type=Path, help="Image path.", required=True)
    parser.add_argument("--dataset-root", type=Path, help="Path to JHU Crowd++ dataset.")
    parser.add_argument("--subset", type=str, default="test", help="Subset.")

    return parser.parse_args()


def main():
    args = get_args()

    if args.ckpt_path:
        model = LitCrowdNet.load_from_checkpoint(args.ckpt_path, map_location=args.device)
        model.eval()
    else:
        model = torch.load(args.model_path, map_location=args.device)
        model.eval()

    img = load_image(args.img_path)
    img, resize_ratio = resize_img(img)

    img_input = preprocess_transform(img).unsqueeze(dim=0)
    pred_density_map = model(img_input).detach().cpu().numpy()[0][0]

    pred_count = count_from_density_map(pred_density_map)
    print("Predicted number of people:", pred_count)

    if args.dataset_root is not None:
        dataset = JHUCrowdDataset(args.dataset_root, subset_name=args.subset)
        print("---------------Ground truth---------------")
        print(dataset.img_labels_df.loc[str(args.img_path.stem)])

    plt.imshow(img)
    plt.imshow(cv2.resize(pred_density_map, None, fx=8, fy=8), cmap="jet", alpha=0.4)
    plt.show()


if __name__ == "__main__":
    main()
