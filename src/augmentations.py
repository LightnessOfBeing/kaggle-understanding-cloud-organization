import cv2

import albumentations as albu

from src.utils import to_tensor


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_transforms(aug_name, image_size=None):
    OPTIONS = {
        'train': get_training_augmentation2,
        'valid': get_validation_augmentation
    }
    augs = OPTIONS[aug_name]()
    if image_size is not None:
        augs = [albu.Resize(*image_size)] + augs
    return albu.Compose(augs)


def get_training_augmentation0():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.5),
        albu.RandomGamma(),
    ]
    return train_transform


def get_training_augmentation1():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2)
    ]
    return train_transform


def get_training_augmentation2():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.RandomBrightnessContrast(p=0.5)
    ]
    return train_transform


def get_training_augmentation_both():
    train_transform = [
        albu.VerticalFlip(p=0.6),
        albu.HorizontalFlip(p=0.6),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.6, border_mode=0),
        albu.GridDistortion(p=0.6),
        albu.OpticalDistortion(p=0.6, distort_limit=0.1, shift_limit=0.2),
        albu.RandomBrightnessContrast(p=0.6)
    ]
    return train_transform


def get_training_augmentation_none():
    train_transform = [
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.6, border_mode=0),
        albu.GridDistortion(p=0.6),
        albu.OpticalDistortion(p=0.6, distort_limit=0.1, shift_limit=0.2),
        albu.RandomBrightnessContrast(p=0.6)
    ]
    return train_transform


def get_training_augmentation3():
    train_transform = [
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                              rotate_limit=15,
                              border_mode=cv2.BORDER_CONSTANT, value=0),
        albu.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=0),
        albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        albu.Solarize(threshold=128, always_apply=False, p=0.5),
        albu.HorizontalFlip(p=0.5)
    ]
    return train_transform


def get_validation_augmentation():
    return albu.Compose([])
