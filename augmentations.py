import albumentations as albu

def to_tensor(x):
    return x.transpose(2, 0, 1).astype('float32')


def get_training_augmentation(augmentation: str = 'default', image_size: tuple = (320, 640)):
    LEVELS = {
        'default': get_training_augmentation0,
        '1': get_training_augmentation1,
        '2': get_training_augmentation2
    }

    assert augmentation in LEVELS.keys()
    return LEVELS[augmentation]()


def get_training_augmentation0():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.5),
        albu.RandomGamma()
    ]
    return albu.Compose(train_transform)


def get_training_augmentation1():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
    ]
    return albu.Compose(train_transform)


def get_training_augmentation2():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.Blur(),
        albu.RandomBrightnessContrast()
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(image_size: tuple = (320, 640)):
    test_transform = [
       # albu.Resize(*image_size)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
