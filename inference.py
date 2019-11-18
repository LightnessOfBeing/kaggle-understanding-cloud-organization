import datetime
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
from models import get_model
from utils import post_process, draw_convex_hull


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(loaders=None,
            runner=None,
            class_params: dict = None,
            path: str = '',
            sub_name: str = '',
            convex_hull: bool = False,
            add_name: str = ""):
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
    pred_distr = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
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
                    pred_distr[-1] += 1
                    encoded_pixels.append('')
                    if convex_hull:
                        encoded_pixels_ch.append('')
                else:
                    pred_distr[image_id % 4] += 1
                    r = mask2rle(prediction)
                    encoded_pixels.append(r)
                    if convex_hull:
                        r_ch = mask2rle(draw_convex_hull(prediction.astype(np.uint8)))
                        encoded_pixels_ch.append(r_ch)
                image_id += 1

    print(f"empty={pred_distr[-1]} fish={pred_distr[0]} flower={pred_distr[1]} gravel={pred_distr[2]} sugar={pred_distr[3]}")

    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'submission_{sub_name}{add_name}.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
    if convex_hull:
        sub_ch = sub.copy()
        sub_ch["EncodedPixels"] = encoded_pixels_ch
        sub_ch.to_csv(f'submission_{sub_name}{add_name}_ch.csv', columns=['Image_Label', 'EncodedPixels'], index=False)


def get_class_params(json_path=None, mode="custom"):
    if mode == "custom":
        assert ".json" in json_path
        json_file = open(json_path)
        data = json.load(json_file)
        class_params = {0: data["0"], 1: data["1"], 2: data["2"], 3: data["3"]}
        return class_params
    elif mode == "average":
        assert ".json" not in json_path
        files = sorted(list(filter(lambda x: "json" in x, os.listdir(json_path))))
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
        assert ".json" not in json_path
        class_params_arr = []
        files = sorted(list(filter(lambda x: "json" in x, os.listdir(json_path))))
        print(f"Json files {files}")
        for file_path in files:
            json_file = open(os.path.join(json_path, file_path))
            data = json.load(json_file)
            class_params_arr.append({0: data["0"], 1: data["1"], 2: data["2"], 3: data["3"]})
        return class_params_arr

def get_encoder_names(weights_names):
    return [name.split("_enc_")[0] for name in weights_names]

def get_thresholds(threshold_mode, json_path):
    if threshold_mode == "all":
        class_params_arr = get_class_params(json_path=json_path, mode="all")
        return class_params_arr
    elif threshold_mode == "simple":
        threshold = 0.5
        mask_size = 5000
    elif threshold_mode == "custom":
        threshold, mask_size = get_class_params(json_path=json_path, mode="custom")
    elif threshold_mode == "average":
        class_params_arr = get_class_params(json_path=json_path, mode="average")
        threshold = 0
        mask_size = 0
        for t, ms in class_params_arr:
            threshold += t
            mask_size += ms
        threshold /= len(class_params_arr)
        mask_size /= len(class_params_arr)
    return threshold, mask_size


def get_ensemble_prediction(loaders, weights_path, technique="voting", threshold_mode="all", json_path="./ensemble", path="." ,convex_hull=True):
    if technique == "averaging" and threshold_mode == "all":
        raise ValueError(f'technique={technique} and threshold_mode={threshold_mode} cannot be combined')

    if threshold_mode == "all":
        print("getting class_params")
        class_params_arr = get_thresholds(threshold_mode, json_path)
        print(f"class_params_arr={class_params_arr}")
    else:
        threshold, mask_size = get_thresholds(threshold_mode, json_path)

    weights_names = sorted(list(filter(lambda x: "pth" in x, os.listdir(weights_path))))
    print(f"weights_names {weights_names}")
    num_models = len(weights_names)
    print(f"Num models={num_models}")
    models = [None] * num_models
    runners = [None] * num_models
    encoder_names = get_encoder_names(weights_names)

    print("loading weights")
    for i in range(num_models):
        models[i] = get_model(model_type="Unet", encoder=encoder_names[i],
                              encoder_weights="imagenet",
                              activation=None, task="segmentation")
        print(f"enc={encoder_names[i]} weight={weights_names[i]}")
        checkpoint = utils.load_checkpoint(os.path.join(weights_path, weights_names[i]))
        models[i].cuda()
        utils.unpack_checkpoint(checkpoint, model=models[i])
        runners[i] = SupervisedRunner(model=models[i])
        loader_valid = {"infer": loaders["valid"]}
        runners[i].infer(
            model=models[i],
            loaders=loader_valid,
            callbacks=[
                InferCallback()
            ],
            verbose=True
        )
    iters = 0
    pred_distr = {-1:0, 0:0, 1:0, 2:0, 3:0}

    encoded_pixels = []
    encoded_pixels_ch = []

    if technique == "averaging":
        print("Technique = averaging")
        for _, test_batch in enumerate(tqdm.tqdm(loaders['test'])):
            runner_out_arr = [runners[i].predict_batch({"features": test_batch[0].cuda()})['logits']
                              for i in range(len(runners))]
            runner_out_len = len(runner_out_arr)
            assert runner_out_len == num_models
            batch_len = len(runner_out_arr[0])
            pred_len = len(runner_out_arr[0][0])
            for batch_id in range(batch_len):
                for pred_id in range(pred_len):
                    probability_final = np.zeros((350, 525))
                    for run_id in range(runner_out_len):
                        probability_model = runner_out_arr[run_id][batch_id][pred_id].cpu().detach().numpy()
                        if probability_model.shape != (350, 525):
                            probability_model = cv2.resize(probability_model, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                        probability_final += sigmoid(probability_model)
                    probability_final /= num_models
                   # print(f"probability final = {probability_final}")
                    prediction, num_predict = post_process(probability_final, threshold, mask_size)
                   # print(f"prediction = {prediction}")
                    if num_predict == 0:
                        pred_distr[-1] += 1
                        encoded_pixels.append('')
                        if convex_hull:
                            encoded_pixels_ch.append('')
                    else:
                        pred_distr[iters % 4] += 1
                        r = mask2rle(prediction)
                        encoded_pixels.append(r)
                        if convex_hull:
                            r_ch = mask2rle(draw_convex_hull(prediction.astype(np.uint8)))
                            encoded_pixels_ch.append(r_ch)
                    iters += 1
    elif technique == "voting":
        print("Technique = voting")
        threshold = num_models // 2 + 1
        print(f"threshold = {threshold}")
        for _, test_batch in enumerate(tqdm.tqdm(loaders['test'])):
            runner_out_arr = [runners[i].predict_batch({"features": test_batch[0].cuda()})['logits']
                              for i in range(len(runners))]
            runner_out_len = len(runner_out_arr)
            batch_len = len(runner_out_arr[0])
            pred_len = len(runner_out_arr[0][0])
            for batch_id in range(batch_len):
                for pred_id in range(pred_len):
                    prediction_final = np.zeros((350, 525))
                    for run_id in range(runner_out_len):
                        probability_model = runner_out_arr[run_id][batch_id][pred_id].cpu().detach().numpy()
                        if probability_model.shape != (350, 525):
                            probability_model = cv2.resize(probability_model, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                        if threshold_mode == "all":
                            print("kek")
                            prediction_model, num_predict = post_process(sigmoid(probability_model),
                                                               class_params_arr[run_id][iters % 4][0],
                                                               class_params_arr[run_id][iters % 4][1])
                        else:
                            prediction_model, num_predict = post_process(sigmoid(probability_model),
                                                                         0.5,
                                                                         5000)
                        prediction_final += prediction_model

                    prediction_final = np.where(prediction_final >= threshold, 1, 0)
                    if prediction_final.sum() == 0:
                    #if num_predict == 0:
                        pred_distr[-1] += 1
                        encoded_pixels.append('')
                        if convex_hull:
                            encoded_pixels_ch.append('')
                    else:
                        pred_distr[iters % 4] += 1
                        r = mask2rle(prediction_final)
                        encoded_pixels.append(r)
                        if convex_hull:
                            r_ch = mask2rle(draw_convex_hull(prediction_final.astype(np.uint8)))
                            encoded_pixels_ch.append(r_ch)
                    iters += 1

    print(iters)
    assert  iters == 14792
    # {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    print(f"empty={pred_distr[-1]} fish={pred_distr[0]} flower={pred_distr[1]} gravel={pred_distr[2]} sugar={pred_distr[3]}")
    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['EncodedPixels'] = encoded_pixels
    sub_name = f'{str(datetime.datetime.now().date())}'
    sub.to_csv(f'submission_{sub_name}_{technique}_ens.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
    if convex_hull:
        sub_ch = sub.copy()
        sub_ch["EncodedPixels"] = encoded_pixels_ch
        sub_ch.to_csv(f'submission_{sub_name}_{technique}_ch_ens.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
