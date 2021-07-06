
# Kaggle Understanding cloud organization
https://www.kaggle.com/c/understanding_cloud_organization

**95th place solution (out of 1,538 teams Top 6.2%)**

**private/public score**: 0.65800/0.67007 (Sørensen–Dice coefficient)

## Competition description

<img src="https://storage.googleapis.com/kaggle-media/competitions/MaxPlanck/Teaser_AnimationwLabels.gif" width="60%" height="60%" align="center">

You need to classify cloud organization patterns from satellite images. If successful, you’ll help scientists to better understand how clouds will shape our future climate. This research will guide the development of next-generation models which could reduce uncertainties in climate projections.

## Poster 
[Link](https://github.com/LightnessOfBeing/kaggle-understanding-cloud-organization/blob/master/poster/Poster_Clouds.pdf) to the poster, which describes our approach end-to-end.

This work has been presented at summer/winter schools:  
- [DS<sup>3</sup> 2021](https://www.ds3-datascience-polytechnique.fr/wp-content/uploads/2021/01/10.pdf)  
- [SMILES 2020](https://smiles.skoltech.ru/poster-presentations)  
- [EEML 2020](https://www.eeml.eu/previous-editions/eeml2020)

## Abstract
Climate change has been at the top of our minds and at the forefront of important political decision-making for many years. Classification of different types of clouds is substantial for understanding climate change. Human ability to identify patterns is limited and murky boundaries between different forms of clouds lead to obstacles in traditional rule-based algorithms cloud features separation. In these situations, machine learning techniques, particularly deep learning, have demonstrated their ability to mimic the human capacity for identifying patterns in the clouds using satellite images. This work focuses on the **segmentation of four subjective patterns of clouds** organization: **Sugar**, **Flower**, **Fish**, **Gravel**.

<img align="left" src="https://raw.githubusercontent.com/kabartay/kaggle-understanding-cloud-organization/main/docs/labels_nomasks.png" data-canonical-src="https://raw.githubusercontent.com/kabartay/kaggle-understanding-cloud-organization/main/docs/labels_nomasks.png" width="1000" />

<img align="left" src="https://raw.githubusercontent.com/kabartay/kaggle-understanding-cloud-organization/main/docs/labels_masks.png" data-canonical-src="https://raw.githubusercontent.com/kabartay/kaggle-understanding-cloud-organization/main/docs/labels_masks.png" width="1000" />

&nbsp;
&nbsp;

## Approach

### Preprocessing
1. Removing bad images

### Training phase
Our training phase consists of 2 stages.

#### First stage
  1. Train on 320x640 image resolution with *pseudo-labels*.
  2. Pseudo-labels were generated the following way: pick an image if there are at least 80% of high-confidence pixels, i.e its values are either < 0.2 or > 0.8.
  3. Optimizer: `Adam`, encoder `lr = 1e-4`, decoder `lr = 1e-3`.
  4. Augmentations: we tried a big set of non-geometric augmentations:  
  `Blur`, `CLAHE`, `GaussNoise`, `GaussianBlur`, `HueSaturationValue`, `RGBShift`, `IAAAdditiveGaussianNoise`, `MedianBlur`, `MotionBlur`. We thought these  augmentations could mimic different weather conditions and add different sunlight effects. However, the LB was very bad, so we decided to remove most of the non-geometric augs and went with the following.

  * `albu.HorizontalFlip(p=0.5)`,
  * `albu.VerticalFlip(p=0.5)`,
  * `albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0)`,
  * `albu.GridDistortion(p=0.5)`,
  * `albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2)`,
  * `albu.RandomBrightnessContrast(p=0.5)`

 5. Loss function: `BceDiceLoss` with `eps=10`.
 We grid-searched on eps value and it gave +0.003 LB.

 6. Scheduler: `ReduceLrOnPlateu`, with `patience=2`.

 Typically, models from the first stage were overfitting after 21-22 epochs.

#### Second stage

  0. Upload weights with the best valid loss from the first stage.
  1. Train on 512x768 image resolutions only on train data.
  2. Optimizer: `Adam`, encoder `lr = 1e-4`, decoder `lr = 5e-4`.
  3. Augmentations: same as in the first stage.
  4. Loss function: sum of `Symmetric Lovasz Losses` for each of the channels.
  5. Scheduler: `ReduceLrOnPlateu`

### Postprocessing

We used the *threshold* of 0.5 and removed all masks that were smaller than 5000 pixels.

### Inference and blending

Two days before the competition I read a message on the forum which was indicating the following: as our predictions approach 38.5% of non-empty mask the score exceeds 0.670 public lb. Our best submissions at that moment were generating 36.2 - 36.6% of non-empty masks, so we needed to lower the number of false negatives. At the same time, we were scared of the risk of generating a bigger number of false positives.

We took in our final blend the following 5 submissions:
1. EfficientNet-B0 2 stage last epoch
2. EfficientNet-B0 2 stage best epoch
3. EfficientNet-B2 2 stage best epoch
4. EfficientNet-B5 2 stage best epoch
5. One artificial blend submission:

We needed this artificial submission as we wanted to push % of non-empty masks closer to 38.5%.
 We took submissions from the following single-fold models:
  * B0 2stage last epoch
  * B0 2 stage best epoch
  * B1 2 stage best epoch
  * B5 2 stage best epoch
  * se_resnext50 best epoch.

 Then we performed a voting blend on them with vote threshold t = 2 and generated this submission. This submission produced 38.91% of non-empty masks.

After adding an artificial submission to ensemble the % of non-empty masks rose from 36.5% to 37.45%. After blending the 5 submissions from above with voting threshold t = 3 we got a submission with private/public score of 0.65800/0.67007 (our highest public lb score).

### Why we thought it will work
* Two-stage training with on images with different aspect ratios (320x640 and 512x768).
* Second stage trained only on training data.
* Independent `Symmetric Lovasz Loss` for each channel.
* Adding one artificial submission to lower the number of false negatives, whist keeping threshold high enough to reduce the number of false positives.

### What could be done
* Adding a classifier and lowering a pixel thresold.
* Removing images with 3 and 4 overlapping masks from train.
* Weights averaging between folds.
* Weights averaging between k best scores of a single fold model.
* Ensembling bigger number of models.

## How to run 
You can access original dataset here: https://www.kaggle.com/c/understanding_cloud_organization/data  

To run the pipeline with a configuration file from the project root folder:
```
catalyst-dl run --expdir src --logdir {LOGDIR_NAME} --config {CONFIG_PATH}
```

## Citation

Please use this bibtex if you want to cite this work in your publications:

    @misc{kirill_a_vishniakov_2021_5055450,
      author       = {Kirill A. Vishniakov and Mukharbek Organokov},
      title        = {{Segmentation of cloud patterns from satellite images to improve climate models}},
      month        = jul,
      year         = 2021,
      note         = {{Code is available at GitHub: 
                       https://github.com/LightnessOfBeing/kaggle-understanding-cloud-organization}},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.5055450},
      url          = {https://doi.org/10.5281/zenodo.5055450}
    }

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5055450.svg)](https://doi.org/10.5281/zenodo.5055450)

## Credits

* [Mukharbek Organokov](https://www.kaggle.com/muhakabartay) and my other teammates.
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [albumentations library for augmentations](https://github.com/albumentations-team/albumentations)
* [Andrew Lukyanenko](https://www.kaggle.com/artgor) for sharing [this](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools) kernel and [pipeline](https://github.com/Erlemar/Understanding-Clouds-from-Satellite-Images)
