import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import settings

train_aug = A.Compose([
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5),
    A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
    ToTensorV2(transpose_mask=True)
])
valid_aug = A.Compose([
    A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
    ToTensorV2(transpose_mask=True)
])
