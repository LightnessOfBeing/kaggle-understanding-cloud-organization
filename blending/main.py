#!/usr/bin/env python
import argparse
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_img(x: str = 'img_name', folder: str = 'train_images'):
    """
    Return image based on image name and folder.
    Args:
        x: image name
        folder: folder with images
    Returns:
    """
    image_path = os.path.join(folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = '', shape: tuple = (350, 525)):
    """
    Decode rle encoded mask.
    Args:
        mask_rle: encoded mask
        shape: final shape
    Returns:
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (350, 525)):
    """
    Create mask based on df, image name and shape.
    Args:
        df: dataframe with cloud dataset
        image_name: image name
        shape: final shape
    Returns:
    """

    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def mask2rle(img):
    """
    Convert mask to rle.
    Args:
        img:
    Returns:
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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

