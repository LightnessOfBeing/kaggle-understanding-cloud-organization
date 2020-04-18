import os
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from src.augmentations import get_preprocessing
from src.utils import make_mask, to_tensor


class CloudDataset(Dataset):
    def __init__(self,
                 df,
                 path,
                 img_ids,
                 image_folder,
                 transforms,
                 preprocessing_fn):
        self.df = df
        self.img_ids = img_ids
        self.path = path
        self.data_folder = os.path.join(self.path, image_folder)
        bad_imgs = ['046586a.jpg', '1588d4c.jpg', '1e40a05.jpg',
                    '41f92e5.jpg', '449b792.jpg', '563fc48.jpg',
                    '8bd81ce.jpg', 'c0306e5.jpg', 'c26c635.jpg',
                    'e04fea3.jpg', 'e5f2f24.jpg', 'eda52f2.jpg',
                    'fa645da.jpg', 'b092cc1.jpg', 'ee0ba55.jpg']
        self.img_ids = [i for i in self.img_ids if i not in bad_imgs]
        self.transforms = transforms
        self.preprocessing = get_preprocessing(preprocessing_fn)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    '''
    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.image_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        image = to_tensor(self.preprocessing_fn(augmented['image']))
        mask = to_tensor(augmented['mask'])
        return image, mask
    '''

    def __len__(self):
        return len(self.img_ids)
