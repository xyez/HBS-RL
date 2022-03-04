from __future__ import print_function

import sys

import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    pass
else:
    pass

import torch.utils.data as data


class NUS(data.Dataset):

    def __init__(self, root, data_file, train=True, transform=None,
                 target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        data = np.load(data_file)
        self.labels = np.array(data['arr_0']).astype('float32')
        self.paths = data['arr_1']
        print(self.paths.shape, self.labels.shape, self.transform)
        self.train = True
        self.num = self.labels.shape[0]

    def __getitem__(self, index):
        path = self.root + self.paths[index].strip()
        path = path.replace('\\', '/')
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.fromarray(np.random.rand((224, 224, 3)))
        target = self.labels[index, :]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.labels.shape[0]
