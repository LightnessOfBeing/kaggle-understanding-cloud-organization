#!/usr/bin/env python
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import rle_decode, mask2rle

if __name__ == "__main__":
    '''
    Used for blending submissions
    !python main.py --path my_path
    '''

    parser = argparse.ArgumentParser(description="blend submissions")
    parser.add_argument("--path", help="path to submissions folder", type=str, default='./submissions')
    parser.add_argument("--name", help="suffix", type=str, default=None)
    parser.add_argument("--vote_threshold", type=int, default=None)

    args = parser.parse_args()

    threshold = args.vote_threshold
    filenames = os.listdir(args.path)
    dataframes = [pd.read_csv(os.path.join(args.path, name)) for name in filenames]
    if threshold == None:
        threshold = len(dataframes) // 2 + 1
    test_samples = len(dataframes[0])

    print(f"voting threshold = {threshold}")
    pred_distr = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    encoded_pixels = []
    encoded_pixels_ch = []
    print("processing submissions")
    for i in tqdm(range(test_samples)):
        mask = np.zeros((350, 525))
        for j in range(len(dataframes)):
            if dataframes[j].iloc[i]["EncodedPixels"] is not np.nan:
                mask += rle_decode(dataframes[j].iloc[i]["EncodedPixels"])
        mask_final = np.where(mask >= threshold, 1, 0)
        if mask_final.sum() == 0:
            pred_distr[-1] += 1
            encoded_pixels.append('')
        else:
            pred_distr[i % 4] += 1
            r = mask2rle(mask_final)
            encoded_pixels.append(r)

    print(f"empty={pred_distr[-1]} fish={pred_distr[0]} flower={pred_distr[1]} gravel={pred_distr[2]} sugar={pred_distr[3]}")
    non_empty = pred_distr[0] + pred_distr[1]+ pred_distr[2]+ pred_distr[3]
    total = non_empty + pred_distr[-1]
    print(f"% of non-empty masks {round(non_empty / total, 4)}")

    sub_final = dataframes[0].copy()
    sub_final["EncodedPixels"] = encoded_pixels
    if args.name is None:
        name = ""
    else:
        name = "_" + args.name
    sub_final.to_csv(f"./submission_blend{name}.csv", index=None)

