import argparse
import os
import random

import dataloader
import numpy as np
import torch as t
import torchvision
from misc import progress_bar


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)

    alexnet = torchvision.models.alexnet(pretrained=True)
    alexnet.to(args.device)
    alexnet.eval()

    database_txt = os.path.join(args.root, args.dataset, 'database.txt')
    database = dataloader.ImageList(open(database_txt).readlines(), transform=dataloader.image_test(resize_size=255, crop_size=224))
    database_loader = t.utils.data.DataLoader(dataset=database, batch_size=args.batch_size, shuffle=False)

    res = {'rF': [], 'rL': []}

    with t.no_grad():
        for batch_num, (data, target) in enumerate(database_loader):
            data, target = data.to(args.device), target.to(args.device)
            feature = alexnet.features(data)
            feature = alexnet.avgpool(feature).flatten(1)
            feature = alexnet.classifier[:5](feature)
            feature = feature.view(data.shape[0], -1)

            res['rF'].append(feature)
            res['rL'].append(target)

            progress_bar(batch_num, len(database_loader))

    res['rF'] = t.cat(res['rF'], dim=0).cpu()
    res['rL'] = t.cat(res['rL'], dim=0).cpu()
    file_path = os.path.join('feature', args.dataset)
    os.makedirs(file_path, exist_ok=True)
    t.save(res, os.path.join(file_path, 'feature.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
