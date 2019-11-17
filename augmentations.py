import cv2

import albumentations as albu


def to_tensor(x, **kwargs):
    """
    Convert image or mask.

    Args:
        x:
        **kwargs:

    Returns:

    """

    return x.transpose(2, 0, 1).astype('float32')


def get_training_augmentation(augmentation: str='default', image_size: tuple = (320, 640)):
    """
    Get augmentations
    There is a dictionary where values are different augmentation functions, so it easy to
    switch between augmentations;

    Args:
        augmentation:
        image_size:

    Returns:

    """
    LEVELS = {
        'default': get_training_augmentation0,
        '1': get_training_augmentation1,
        '2': get_training_augmentation2,
        '3': get_training_augmentation3,
        'both': get_training_augmentation_both,
        'none': get_training_augmentation_none
    }

    assert augmentation in LEVELS.keys()
    return LEVELS[augmentation](image_size)


def get_training_augmentation0(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.5),
        albu.RandomGamma(),
        albu.Resize(*image_size)
    ]
    return albu.Compose(train_transform)


def get_training_augmentation1(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.Resize(*image_size),
    ]
    return albu.Compose(train_transform)


def get_training_augmentation2(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
        #albu.Resize(*image_size),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.RandomBrightnessContrast(p=0.5)
    ]
    return albu.Compose(train_transform)

def get_training_augmentation_both(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
      #  albu.Resize(*image_size),
        albu.VerticalFlip(p=0.6),
        albu.HorizontalFlip(p=0.6),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.6, border_mode=0),
        albu.GridDistortion(p=0.6),
        albu.OpticalDistortion(p=0.6, distort_limit=0.1, shift_limit=0.2),
        albu.RandomBrightnessContrast(p=0.6)
    ]
    return albu.Compose(train_transform)

def get_training_augmentation_none(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
      #  albu.Resize(*image_size),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.6, border_mode=0),
        albu.GridDistortion(p=0.6),
        albu.OpticalDistortion(p=0.6, distort_limit=0.1, shift_limit=0.2),
        albu.RandomBrightnessContrast(p=0.6)
    ]
    return albu.Compose(train_transform)


def get_training_augmentation3(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
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
    return albu.Compose(train_transform)


def get_validation_augmentation(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    test_transform = [
       # albu.Resize(*image_size)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
