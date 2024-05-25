#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import os
from torch.utils.data import Dataset
from PIL import Image

class SegData(Dataset):
    def __init__(self, image_path, mask_path, data_transforms=None):
        self.image_path = image_path
        self.mask_path = mask_path

        self.images = os.listdir(self.image_path)
        self.masks = os.listdir(self.mask_path)
        self.transform = data_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filename = self.images[idx]
        mask_filename = image_filename.replace('jpeg','png')

        image = Image.open(os.path.join(self.image_path, image_filename)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path, mask_filename)).convert('L')

        if self.transform is not None:
            image, mask = self.transform(image ,mask)

        return image, mask