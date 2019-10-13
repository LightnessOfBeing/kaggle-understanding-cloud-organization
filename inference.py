import gc
import torch

import cv2
import numpy as np
import pandas as pd
import tqdm

from utils import post_process, draw_convex_hull
from dataset import mask2rle


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
        print("convex hull is used")
        sub_ch = sub.copy()
        sub_ch["EncodedPixels"] = encoded_pixels_ch
        sub_ch.to_csv(f'submission_{sub_name}_ch.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
