from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_ct_img_path = os.path.join(root, 'train', "CT")
        train_pet_img_path = os.path.join(root, 'train', "PET")
        train_mask_path = os.path.join(root, 'train_GT')

        ct_images = os.listdir(train_ct_img_path)
        labels = os.listdir(train_mask_path)
        ct_images.sort()
        labels.sort()

        for it_im, it_gt in zip(ct_images, labels):
            item = (os.path.join(train_ct_img_path, it_im),os.path.join(train_pet_img_path, it_im),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
    elif mode == 'val':
        train_ct_img_path = os.path.join(root, 'valid', "CT")
        train_pet_img_path = os.path.join(root, 'valid', "PET")
        train_mask_path = os.path.join(root, 'valid_GT')

        images = os.listdir(train_ct_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_ct_img_path, it_im), os.path.join(train_pet_img_path, it_im),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
    else:
        train_ct_img_path = os.path.join(root, 'test', "CT")
        train_pet_img_path = os.path.join(root, 'test', "PET")
        train_mask_path = os.path.join(root, 'test_GT')

        images = os.listdir(train_ct_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_ct_img_path, it_im), os.path.join(train_pet_img_path, it_im),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
    # print(items)
    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize

    def __len__(self):
        return len(self.imgs)

    def augment(self, ct_img,pet_img, mask):
        if random() > 0.5:
            ct_img = ImageOps.flip(ct_img)
            pet_img = ImageOps.flip(pet_img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            ct_img = ImageOps.mirror(ct_img)
            pet_img = ImageOps.mirror(pet_img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            ct_img = ct_img.rotate(angle)
            pet_img = pet_img.rotate(angle)
            mask = mask.rotate(angle)
        return ct_img, pet_img, mask

    def __getitem__(self, index):
        ct_img_path, pet_img_path, mask_path = self.imgs[index]
        # print("{} and {}".format(img_path,mask_path))
        # img = Image.open(img_path)  # .convert('RGB')
        # mask = Image.open(mask_path)  # .convert('RGB')
        ct_img = Image.open(ct_img_path).convert('L')
        pet_img = Image.open(pet_img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # print('{} and {}'.format(img_path,mask_path))
        # if self.equalize:
        #     img = ImageOps.equalize(img)

        if self.augmentation:
            ct_img, pet_img, mask = self.augment(ct_img, pet_img, mask)

        if self.transform:
            ct_img = self.transform(ct_img)
            pet_img = self.transform(pet_img)
            mask = self.mask_transform(mask)


        return [ct_img, pet_img, mask]
