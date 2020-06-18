import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import DataLoader

from src.augmentations import get_transforms
from src.dataset import CloudDataset
from src.utils import mask2rle, post_process, sigmoid


def infer(model, testloader, class_params, output_path):
    encoded_pixels = []
    pred_distr = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    image_id = 0
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm.tqdm(testloader, total=len(testloader)):
            masks = model(images)
            for mask in masks:
                mask = mask.cpu().detach().numpy()
                if mask.shape != (350, 525):
                    mask = cv2.resize(
                        mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                    )
                mask, num_predict = post_process(
                    sigmoid(mask),
                    class_params[image_id % 4][0],
                    class_params[image_id % 4][1],
                )
                if num_predict == 0:
                    pred_distr[-1] += 1
                    encoded_pixels.append("")
                else:
                    pred_distr[image_id % 4] += 1
                    r = mask2rle(mask)
                    encoded_pixels.append(r)
                image_id += 1

    print(
        f"empty={pred_distr[-1]} fish={pred_distr[0]} flower={pred_distr[1]} gravel={pred_distr[2]} sugar={pred_distr[3]}"
    )
    non_empty = pred_distr[0] + pred_distr[1] + pred_distr[2] + pred_distr[3]
    all = non_empty + pred_distr[-1]
    sub = pd.read_csv(f"{output_path}/sample_submission.csv")
    sub["EncodedPixels"] = encoded_pixels
    sub.to_csv(
        f"submission_{round(non_empty/all, 3)}.csv",
        columns=["Image_Label", "EncodedPixels"],
        index=False,
    )


if __name__ == "__main__":
    model_name = sys.argv[1]
    test_data_path = sys.argv[2]
    class_params_path = sys.argv[3]
    output_path = sys.argv[3]

    df_test = pd.read_csv(os.path.join(test_data_path), "sample_submission.csv")
    test_ids = (
        df_test["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    )
    preprocess_fn = get_preprocessing_fn(model_name, "imagenet")
    test_dataset = CloudDataset(
        df=df_test,
        path=test_data_path,
        img_ids=test_ids,
        image_folder="test_images",
        transforms=get_transforms("valid"),
        preprocessing_fn=preprocess_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4,
    )
    model = Unet(model_name, classes=4, activation=None)
    class_params = np.load(class_params_path).item()
    infer(model, test_loader, class_params, output_path)
