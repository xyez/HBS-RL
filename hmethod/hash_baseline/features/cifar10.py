import random
import os

import numpy as np
import torch as t
import torchvision
from misc import progress_bar
from torchvision import transforms as transforms


def main():
    seed = 123
    device = 'cuda:0'
    batch_size = 100
    root_path = './hmethod/hash_baseline/features/data_cifar10'

    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(123)

    alexnet = torchvision.models.alexnet(pretrained=True)
    alexnet.to(device)
    alexnet.eval()

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform_test)
    train_loader = t.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    test_set = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform_test)
    test_loader = t.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    res = {'rF': [], 'rL': [], 'qF': [], 'qL': []}

    with t.no_grad():
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            feature = alexnet.features(data)
            feature = alexnet.avgpool(feature).flatten(1)
            feature = alexnet.classifier[:5](feature)
            feature = feature.view(data.shape[0], -1)

            res['rF'].append(feature)
            res['rL'].append(target)

            progress_bar(batch_num, len(train_loader))

    with t.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            feature = alexnet.features(data)
            feature = alexnet.avgpool(feature).flatten(1)
            feature = alexnet.classifier[:5](feature)
            feature = feature.view(data.shape[0], -1)

            res['qF'].append(feature)
            res['qL'].append(target)

            progress_bar(batch_num, len(test_loader))

    res['rF'] = t.cat(res['rF'], dim=0).cpu()
    res['rL'] = t.cat(res['rL'], dim=0).cpu()
    res['qF'] = t.cat(res['qF'], dim=0).cpu()
    res['qL'] = t.cat(res['qL'], dim=0).cpu()

    res['rL'] = t.eye(10)[res['rL']]
    res['qL'] = t.eye(10)[res['qL']]
    os.makedirs('data/feature/cifar10/', exist_ok=True)
    t.save(res, 'data/feature/cifar10/feature.pt')


if __name__ == '__main__':
    main()
