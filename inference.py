import gc
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
import segmentation_models_pytorch as smp
from catalyst import utils
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import InferCallback

from dataset import mask2rle
from utils import post_process, draw_convex_hull


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(loaders=None,
            runner=None,
            class_params: dict = None,
            path: str = '',
            sub_name: str = '',
            convex_hull: bool = False):
    """

    Args:
        loaders:
        runner:
        class_params:
        path:
        sub_name:

    Returns:

    """
    encoded_pixels = []
    if convex_hull:
        print("convex hull is enabled")
        encoded_pixels_ch = []
    image_id = 0
    torch.cuda.empty_cache()
    gc.collect()
    for _, test_batch in tqdm.tqdm(enumerate(loaders['test'])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
        for _, batch in enumerate(runner_out):
            for probability in batch:
                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                    prediction, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
                                                           class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append('')
                    if convex_hull:
                        encoded_pixels_ch.append('')
                else:
                    r = mask2rle(prediction)
                    encoded_pixels.append(r)
                    if convex_hull:
                        r_ch = mask2rle(draw_convex_hull(prediction.astype(np.uint8)))
                        encoded_pixels_ch.append(r_ch)
                image_id += 1

    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'submission_{sub_name}.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
    if convex_hull:
        sub_ch = sub.copy()
        sub_ch["EncodedPixels"] = encoded_pixels_ch
        sub_ch.to_csv(f'submission_{sub_name}_ch.csv', columns=['Image_Label', 'EncodedPixels'], index=False)


def get_class_params(json_folder=None, json_file=None, mode="custom"):
    if mode == "custom":
        json_file = open(json_file)
        data = json.load(json_file)
        class_params = {0: data["0"], 1: data["1"], 2: data["2"], 3: data["3"]}
        return class_params
    elif mode == "average":
        files = sorted(list(filter(lambda x: "json" in x, os.listdir(json_folder))))
        params = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for file_path in files:
            json_file = open(file_path)
            data = json.load(json_file)

            params[0][0] += data["0"][0]
            params[0][1] += data["0"][1]

            params[1][0] += data["1"][0]
            params[1][1] += data["1"][1]

            params[2][0] += data["2"][0]
            params[2][1] += data["2"][1]

            params[3][0] += data["3"][0]
            params[3][1] += data["3"][1]

        return {0: params[0], 1: params[1], 2: params[2], 3: params[3]}
    elif mode == "all":
        class_params_arr = []
        files = sorted(list(filter(lambda x: "json" in x, os.listdir(json_folder))))
        for file_path in files:
            json_file = open(file_path)
            data = json.load(json_file)
            class_params_arr.append({0: data["0"], 1: data["1"], 2: data["2"], 3: data["3"]})
        return class_params_arr

def get_encoder_names(weights_names):
    return [name.split("_enc_")[0] for name in weights_names]

def get_ensemble_prediction(weights_path, loaders):
    weights_names = sorted(list(filter(lambda x: "pth" in x, os.listdir(weights_path))))
    num_ensembles = len(weights_path)
    encoder_names = []
    print("kek")
    predictions_arr = np.zeros((num_ensembles, len(loaders["test"]) * 4, 350, 525), dtype=np.float16)
    print("kek2")
    for i in range(num_ensembles):
        weights_path = weights_path + weights_names[i]
        ENCODER = encoder_names[i]
        ENCODER_WEIGHTS = 'imagenet'
        DEVICE = 'cuda'
        ACTIVATION = None
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=4,
            activation=ACTIVATION,
        )
        checkpoint = utils.load_checkpoint(weights_path)
        model.cuda()
        utils.unpack_checkpoint(checkpoint, model=model)
        runner = SupervisedRunner(model=model)
        loader_valid = {"infer": loaders["valid"]}
        runner.infer(
            model=model,
            loaders=loader_valid,
            callbacks=[
                InferCallback()
            ],
            verbose=False
        )
        iters = 0
        for _, test_batch in enumerate(tqdm.tqdm_notebook(loaders['test'])):
            runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
            for _, batch in enumerate(runner_out):
                for probability in batch:
                    probability = probability.cpu().detach().numpy()
                    if probability.shape != (350, 525):
                        probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                    prediction = sigmoid(probability)
                    predictions_arr[i, iters, :, :] = prediction
                    iters += 1
        assert iters == 14792
    return predictions_arr


def aggregate_ensemble_predictions(ensemble_predictions, mode, json_file=None,
                                   json_folder=None, technique="averaging", convex_hull=True):
    # params "default", "average"

    encoded_pixels = []
    encoded_pixels_ch = []

    threshold = 0
    mask_size = 0
    num_models = ensemble_predictions.shape[0]
    if mode == "simple":
        threshold = 0.5
        mask_size = 20000
    elif mode == "custom":
        threshold, mask_size = get_class_params(json_file=json_file, mode="custom")
    elif mode == "all":
        class_params_arr = get_class_params(json_folder=json_folder, mode="all")
    elif mode == "average":
        class_params_arr = get_class_params(mode="average")
        threshold = 0
        mask_size = 0
        for t, ms in class_params_arr:
            threshold += t
            mask_size += ms
        threshold /= len(class_params_arr)
        mask_size /= len(class_params_arr)

    num_preds = ensemble_predictions.shape[1]
    if technique == "averaging":
        for i in range(num_preds):
            probability_final = np.zeros((350, 525))
            for j in range(num_models):
                probability_model = ensemble_predictions[j, i, :, :]
                probability_final += sigmoid(probability_model)
            probability_final /= num_models
            prediction, num_predict = post_process(sigmoid(probability_final), threshold, mask_size)
            if num_predict == 0:
                encoded_pixels.append('')
                if convex_hull:
                    encoded_pixels_ch.append('')
            else:
                r = mask2rle(prediction)
                encoded_pixels.append(r)
                if convex_hull:
                    r_ch = mask2rle(draw_convex_hull(prediction.astype(np.uint8)))
                    encoded_pixels_ch.append(r_ch)
    elif mode == "voting":
        threshold = num_models // 2 + 1
        iters = 0
        for i in range(num_preds):
            prediction_final = np.zeros((350, 525), dtype=np.int8)
            for j in range(num_models):
                probability_model = ensemble_predictions[j, i, :, :]
                if mode != "all":
                    prediction, num_predict = post_process(sigmoid(probability_model), threshold, mask_size)
                else:
                    prediction, num_predict = post_process(sigmoid(probability_model),
                                                           class_params_arr[j][iters % 4][0],
                                                           class_params_arr[j][iters % 4][1])
                prediction_final += prediction
            iters += 1

            prediction_final[prediction_final >= threshold] = 1
            prediction_final[prediction_final < threshold] = 0
            if prediction_final.sum() == 0:
                encoded_pixels.append('')
                if convex_hull:
                    encoded_pixels_ch.append('')
            else:
                r = mask2rle(prediction_final)
                encoded_pixels.append(r)
                if convex_hull:
                    r_ch = mask2rle(draw_convex_hull(prediction_final.astype(np.uint8)))
                    encoded_pixels_ch.append(r_ch)
    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'submission_{sub_name}_ens.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
    if convex_hull:
        sub_ch = sub.copy()
        sub_ch["EncodedPixels"] = encoded_pixels_ch
        sub_ch.to_csv(f'submission_{sub_name}_ch_ens.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
