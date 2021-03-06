from torch.utils import data
import numpy as np
import torch
import os
from skimage.io import imread
from scipy.io import loadmat
import torchvision.transforms.functional as FT
from haven import haven_utils as hu


import cv2, os
import pandas as pd
import numpy as np
import random
import pandas as pd
import  numpy as np
import os
import torch.utils.data as torchdata

import torch 
import torchvision.transforms.functional as FT
import copy

from torchvision import transforms
import collections
import torch
import numpy as np
import random
from PIL import ImageOps


def get_dataset(dataset_dict, split, datadir, exp_dict, dataset_size=None):
    name = dataset_dict['name']

    if name == 'trancos':
        dataset = Trancos(split, datadir=datadir, exp_dict=exp_dict)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.img_names = dataset.img_names[:dataset_size[split]]

    elif name == 'shanghai':
        dataset = shanghai.Shanghai(split, datadir=datadir, exp_dict=exp_dict)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.img_names = dataset.img_names[:dataset_size[split]]


    else:
        raise ValueError('dataset %s not defined.' % name)

    return dataset


class Trancos(data.Dataset):
    def __init__(self, split, datadir, exp_dict):
        self.split = split
        self.exp_dict = exp_dict
        
        self.n_classes = 1
        
        if split == "train":
            fname = os.path.join(datadir, 'image_sets', 'training.txt')

        elif split == "val":
            fname = os.path.join(datadir, 'image_sets', 'validation.txt')

        elif split == "test":
            fname = os.path.join(datadir, 'image_sets', 'test.txt')

        self.img_names = [name.replace(".jpg\n","") for name in hu.read_text(fname)]
        self.path = os.path.join(datadir, 'images')
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image = imread(os.path.join(self.path, name + ".jpg"))
        points = imread(os.path.join(self.path, name + "dots.png"))[:,:,:1].clip(0,1)
        roi = loadmat(os.path.join(self.path, name + "mask.mat"))["BW"][:,:,np.newaxis]
        
        # LOAD IMG AND POINT
        image = image * roi
        image = hu.shrink2roi(image, roi)
        points = hu.shrink2roi(points, roi).astype("uint8")

        counts = torch.LongTensor(np.array([int(points.sum())]))   
        
        collection = list(map(FT.to_pil_image, [image, points]))
        image, points = apply_transform(self.split, image, points, 
                   transform_name=self.exp_dict['dataset']['transform'])
            
        return {"images":image, 
                "points":points.squeeze(), 
                "counts":counts, 
                'meta':{"index":index}}

from torchvision import transforms
import collections
import torch
import numpy as np


def apply_transform(split, image, points, 
                    transform_name='basic',
                    exp_dict=None):

    if transform_name == 'rgb_normalize':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = ComposeJoint(
            [
                [transforms.ToTensor(), None],
                [transforms.Normalize(mean=mean, std=std), None],
                [None, ToLong()]
            ])

        return transform([image, points])

class ComposeJoint(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = self._iterate_transforms(transform, x)

        return x

    def _iterate_transforms(self, transforms, x):
        if isinstance(transforms, collections.Iterable):
            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:

            if transforms is not None:
                x = transforms(x)

        return x

class ToLong(object):
    def __call__(self, x):
        return torch.LongTensor(np.asarray(x))