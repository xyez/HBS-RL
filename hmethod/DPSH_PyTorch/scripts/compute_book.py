import argparse
import os
import random
import numpy as np
import torch as t

import dataloader
from misc import progress_bar
from models.model_loader import load_model


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)

    database_txt = os.path.join(args.root, args.dataset, 'database.txt')
    database = dataloader.ImageList(open(database_txt).readlines(),
                                    transform=dataloader.image_test(resize_size=255, crop_size=224))
    database_loader = t.utils.data.DataLoader(dataset=database, batch_size=args.batch_size, shuffle=False)

    query_txt = os.path.join(args.root, args.dataset, 'query.txt')
    query = dataloader.ImageList(open(query_txt).readlines(),
                                 transform=dataloader.image_test(resize_size=255, crop_size=224))
    query_loader = t.utils.data.DataLoader(dataset=query, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.arch, args.num_bit).to(args.device)
    checkpoint = t.load(args.model, map_location=args.device)
    with t.no_grad():
        model.load_state_dict(checkpoint['model'])
    model.eval()
    res = {'rB': [], 'rL': [], 'qB': [], 'qL': []}

    with t.no_grad():
        for batch_num, (data, target) in enumerate(database_loader):
            data, target = data.to(args.device), target.to(args.device)
            hash_code = model(data).sign().clamp_min(0)
            res['rB'].append(hash_code.cpu())
            res['rL'].append(target.cpu())
            progress_bar(batch_num, len(database_loader))
    with t.no_grad():
        for batch_num, (data, target) in enumerate(query_loader):
            data, target = data.to(args.device), target.to(args.device)
            hash_code = model(data).sign().clamp_min(0)
            res['qB'].append(hash_code.cpu())
            res['qL'].append(target.cpu())
            progress_bar(batch_num, len(query_loader))

    res['rB'] = t.cat(res['rB'], dim=0).cpu()
    res['rL'] = t.cat(res['rL'], dim=0).cpu()
    res['qB'] = t.cat(res['qB'], dim=0).cpu()
    res['qL'] = t.cat(res['qL'], dim=0).cpu()
    file_path = os.path.join('book', args.dataset, str(args.num_bit))
    os.makedirs(file_path, exist_ok=True)
    t.save(res, os.path.join(file_path, 'book_{}.pt'.format(args.num_bit)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='alexnet', type=str, help='CNN model name.(default: alexnet)')
    parser.add_argument('--model', type=str, default='abc.pt')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--num_bit', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--root', type=str, default='./dataset')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    main(args)
