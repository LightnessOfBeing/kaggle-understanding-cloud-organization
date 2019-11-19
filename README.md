
# Kaggle Understanding cloud organization
https://www.kaggle.com/c/understanding_cloud_organization

95th place solution

## Preprocessing
1. Removing bad images

## Training phase
Our training phase consists of 2 stages.

### First stage
1. Train on 320x640 image resolution with pseudo-labels.
2. Pseudo-labels which were generated the following way: pick an image if there are at least 80% of high-confidance pixels, i.e its values are either < 0.2 or > 0.8. 
3. Optimizer Adam encoder lr = 1e-4, decoder lr = 1e-3
4. Augmentaions: 
  * albu.HorizontalFlip(p=0.5),
  * albu.VerticalFlip(p=0.5),
  * albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
  * albu.GridDistortion(p=0.5),
  * albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
  * albu.RandomBrightnessContrast(p=0.5)
 5. Loss function BceDiceLoss with eps=10.

### Second stage
0. Upload weights with best valid loss from the first stage. 
1. Train on 512x768 image resolutions only on train data.
2. Optimizer Adam encoder lr = 1e-4, decoder lr = 5e-4
3. Same augmentaions from the first stage.
4. Loss function sum of Symmetric Lovasz Losses for each of the channels

## Inference and blending
