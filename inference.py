import gc
import torch

import cv2
import numpy as np
import pandas as pd
import tqdm

from utils import post_process
from dataset import mask2rle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(test_loader=None,
            runner=None,
            class_params: dict = None,
            path: str = '',
            sub_name: str = ''):
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
    image_id = 0
    torch.cuda.empty_cache()
    gc.collect()
    for _, test_batch in tqdm.tqdm(enumerate(test_loader)):
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
                else:
                    r = mask2rle(prediction)
                    encoded_pixels.append(r)
                image_id += 1

    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'submission_{sub_name}.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
