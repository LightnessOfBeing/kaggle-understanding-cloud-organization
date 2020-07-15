import collections
import os

import pandas as pd
from catalyst.dl import ConfigExperiment
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.model_selection import train_test_split

from src.augmentations import get_transforms
from src.dataset import CloudDataset


class Experiment(ConfigExperiment):
    def get_datasets(self, **kwargs):
        path = kwargs.get("path", None)
        df_train_name = kwargs.get("df_train_name", None)
        df_pl_name = kwargs.get("df_pl_name", None)
        image_folder = kwargs.get("image_folder", None)
        encoder_name = kwargs.get("model_name", None)
        test_mode = kwargs.get("test_mode", None)
        type = kwargs.get("type", None)
        height = kwargs.get("height", None)
        width = kwargs.get("width", None)

        if type == "train":
            df_train = pd.read_csv(os.path.join(path, df_train_name))
            if df_pl_name is not None:
                df_pl = pd.read_csv(os.path.join(path, df_pl_name))
                df_train = df_train.append(df_pl)
                print("Pseudo-labels added to train df")

            if test_mode:
                df_train = df_train[:150]

            df_train["label"] = df_train["Image_Label"].apply(lambda x: x.split("_")[1])
            df_train["im_id"] = df_train["Image_Label"].apply(lambda x: x.split("_")[0])

            id_mask_count = (
                df_train.loc[~df_train["EncodedPixels"].isnull(), "Image_Label"]
                    .apply(lambda x: x.split("_")[0])
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "img_id", "Image_Label": "count"})
                    .sort_values(["count", "img_id"])
            )
            assert len(id_mask_count["img_id"].values) == len(
                id_mask_count["img_id"].unique()
            )
            train_ids, valid_ids = train_test_split(
                id_mask_count["img_id"].values,
                random_state=42,
                stratify=id_mask_count["count"],
                test_size=0.1,
            )

        df_test = pd.read_csv(os.path.join(path, "sample_submission.csv"))
        df_test["label"] = df_test["Image_Label"].apply(lambda x: x.split("_")[1])
        df_test["im_id"] = df_test["Image_Label"].apply(lambda x: x.split("_")[0])
        test_ids = (
            df_test["Image_Label"]
                .apply(lambda x: x.split("_")[0])
                .drop_duplicates()
                .values
        )

        preprocess_fn = get_preprocessing_fn(encoder_name, pretrained="imagenet")

        if type != "test":
            train_dataset = CloudDataset(
                df=df_train,
                path=path,
                img_ids=train_ids,
                image_folder=image_folder,
                transforms=get_transforms("train"),
                preprocessing_fn=preprocess_fn,
                height=height,
                width=width
            )

            valid_dataset = CloudDataset(
                df=df_train,
                path=path,
                img_ids=valid_ids,
                image_folder=image_folder,
                transforms=get_transforms("valid"),
                preprocessing_fn=preprocess_fn,
                height=height,
                width=width
            )

        test_dataset = CloudDataset(
            df=df_test,
            path=path,
            img_ids=test_ids,
            image_folder="test_images",
            transforms=get_transforms("valid"),
            preprocessing_fn=preprocess_fn,
            height=height,
            width=width
        )

        datasets = collections.OrderedDict()
        if type == "train":
            datasets["train"] = train_dataset
            datasets["valid"] = valid_dataset
        elif type == "postprocess":
            datasets["infer"] = valid_dataset
        elif type == "test":
            datasets["infer"] = test_dataset

        return datasets
