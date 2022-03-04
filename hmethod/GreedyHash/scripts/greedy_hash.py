import argparse
import os

import numpy as np
import torch as t
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image
from cal_map import get_results
from torch.autograd import Function
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True, database_bool=False, num_classes=100):
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.base_folder = 'train.txt'
        elif database_bool:
            self.base_folder = 'database.txt'
        else:
            self.base_folder = 'test.txt'
        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.base_folder)
        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                pos_tmp = lines.split()[0]
                pos_tmp = os.path.join(self.root, pos_tmp)
                label_tmp = lines.split()[1:]
                self.train_data.append(pos_tmp)
                self.train_labels.append(label_tmp)
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float32)
        self.train_labels.reshape((-1, num_classes))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        target = int(np.where(target == 1)[0])

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


class NusWide(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True, database_bool=False, num_classes=100):
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.base_folder = 'train.txt'
        elif database_bool:
            self.base_folder = 'database.txt'
        else:
            self.base_folder = 'test.txt'
        self.train_data = []
        self.train_labels = []
        filename = os.path.join(self.root, self.base_folder)
        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                pos_tmp = lines.split()[0]
                pos_tmp = os.path.join(self.root, pos_tmp)
                label_tmp = lines.split()[1:]
                self.train_data.append(pos_tmp)
                self.train_labels.append(label_tmp)
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float32)
        self.train_labels.reshape((-1, num_classes))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        img = self.loader(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


class HashLayer(Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, y):
        return y


def hash_layer(x):
    return HashLayer.apply(x)


class CNN(nn.Module):
    def __init__(self, encode_length, num_classes, dataset='imagenet'):
        super(CNN, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(4096, encode_length)
        self.fc = nn.Linear(encode_length, num_classes, bias=False)
        self.is_sigmoid = dataset == 'nuswide'

    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.fc_plus(x)
        code = hash_layer(x)
        output = self.fc(code)
        if self.is_sigmoid:
            output = t.sigmoid(output)

        return output, x, code


def get_lr_adjuster(learning_rate, epoch_lr_decrease):
    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return adjust_learning_rate


def train(args):
    # transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if args.dataset == 'imagenet':
        # Dataset
        Dataset = ImageNet
        criterion = nn.CrossEntropyLoss()
        weight_decay = 5e-4
        label_dtype = t.int64
        loss1_coef = 1.0
        loss2_coef = 1.0
    elif args.dataset == 'nuswide':
        Dataset = NusWide
        criterion = nn.BCELoss()
        # weight_decay = 5e-5
        label_dtype = t.float32
        # loss1_coef = 3.0
        # loss2_coef = 0.1

        # args.num_epochs = 50
        # args.batch_size = 50
        # args.epoch_lr_decrease = 30
        args.lr = 0.001  # 0.001
        args.num_classes = 21

        weight_decay = 5e-4
        loss1_coef = 3.0
        loss2_coef = 0.1
    else:
        raise NotImplementedError()
    print(args)
    train_dataset = Dataset(root=args.root,
                            train=True,
                            transform=train_transform,
                            num_classes=args.num_classes)

    test_dataset = Dataset(root=args.root,
                           train=False,
                           transform=test_transform,
                           num_classes=args.num_classes)

    database_dataset = Dataset(root=args.root,
                               train=False,
                               transform=test_transform,
                               database_bool=True,
                               num_classes=args.num_classes)
    # Data Loader
    train_loader = t.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_cpus)

    test_loader = t.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_cpus)

    database_loader = t.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_cpus)

    net = CNN(encode_length=args.num_bits, num_classes=args.num_classes, dataset=args.dataset)
    net.to(args.device)

    criterion.to(args.device)
    optimizer = t.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
    adjust_learning_rate = get_lr_adjuster(args.lr, args.epoch_lr_decrease)
    best = 0

    for epoch in range(args.num_epochs):
        net.train()
        adjust_learning_rate(optimizer, epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = t.as_tensor(images, dtype=t.float32, device=args.device)
            labels = t.as_tensor(labels, dtype=label_dtype, device=args.device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs, feature, _ = net(images)
            loss1 = criterion(outputs, labels)
            loss2 = t.mean(t.abs(t.pow(t.abs(feature) - t.ones(feature.size(), device=args.device), 3)))
            loss = loss1_coef * loss1 + loss2_coef * loss2
            loss.backward()
            optimizer.step()

            if (i + 1) % (len(train_dataset) // args.batch_size / 2) == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, len(train_dataset) // args.batch_size,
                         loss1.data.item(), loss2.item()))
        torch.save(net.state_dict(), 'data/{}_{}.pkl'.format(args.dataset, args.num_bits))
        net.eval()
        if args.dataset == 'imagenet':
            correct = 0
            total = 0
            with t.no_grad():
                for images, labels in test_loader:
                    images = t.as_tensor(images, dtype=t.float32, device=args.device)
                    outputs, _, _ = net(images)
                    _, predicted = t.max(outputs.cpu().data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

            print('Test Accuracy of the model: %.2f %%' % (100.0 * correct / total))

            if 1.0 * correct / total > best:
                best = 1.0 * correct / total
                t.save(net.state_dict(), 'data/imagenet_{}_best.pkl'.format(args.num_bits))

            print('best: %.2f %%' % (best * 100.0))
        elif args.dataset == 'nuswide':
            if (epoch + 1) % 10 == 0:
                books = {'rB': [],
                         'qB': [],
                         'rL': [],
                         'qL': []}
                for batch_step, (data, target) in enumerate(database_loader):
                    # print(batch_step)
                    data = t.as_tensor(data, dtype=t.float32, device=args.device)
                    _, _, code = net(data)
                    books['rB'] += code.cpu().detach().numpy().tolist()
                    books['rL'] += target.cpu().detach().numpy().tolist()

                for batch_step, (data, target) in enumerate(test_loader):
                    data = t.as_tensor(data, dtype=t.float32, device=args.device)
                    _, _, code = net(data)
                    books['qB'] += code.cpu().detach().numpy().tolist()
                    books['qL'] += target.cpu().detach().numpy().tolist()

                for k in books.keys():
                    books[k] = t.as_tensor(books[k], dtype=t.float32)

                print('---calculate map---')
                result = get_results({'nuswide': books}, args.device, topK=50000)
                print(result['nuswide']['map'])
        else:
            raise NotImplementedError()

    print('over')


def test(args):
    # transform
    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Dataset
    if args.dataset == 'imagenet':
        # Dataset
        Dataset = ImageNet
        label_dtype = t.int64
    elif args.dataset == 'nuswide':
        Dataset = NusWide
        label_dtype = t.float32
        args.num_classes = 21
    else:
        raise NotImplementedError()
    test_dataset = Dataset(root=args.root,
                           train=False,
                           transform=test_transform,
                           num_classes=args.num_classes)

    database_dataset = Dataset(root=args.root,
                               train=False,
                               transform=test_transform,
                               database_bool=True,
                               num_classes=args.num_classes)
    # Data Loader
    test_loader = t.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.num_cpus)

    database_loader = t.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_cpus)

    net = CNN(encode_length=args.num_bits, num_classes=args.num_classes, dataset=args.dataset)
    net.to(args.device)
    net.eval()
    # model_file = os.path.join('models', '{}_{}_best.pkl'.format(args.dataset, args.num_bits))
    model_file = os.path.join('models', '{}_{}.pkl'.format(args.dataset, args.num_bits))
    state_dict = t.load(model_file)
    net.load_state_dict(state_dict)

    qB = []
    qL = []
    for i, (images, labels) in enumerate(test_loader):
        # print(images.shape,images.dtype)
        images = t.as_tensor(images.to(args.device), dtype=t.float32, device=args.device)
        labels = t.as_tensor(labels.to(args.device), dtype=label_dtype, device=args.device)
        if args.dataset == 'imagenet':
            hash_label = t.zeros((labels.size(0), 100), dtype=t.float32, device=args.device)
            hash_label.scatter_(1, labels.unsqueeze(-1), 1)
        else:
            hash_label = labels

        with t.no_grad():
            _, _, hash_code = net(images)

        qB.append(hash_code.float().cpu().numpy())
        qL.append(hash_label.float().cpu().numpy())
    qB = np.concatenate(qB, axis=0)
    qL = np.concatenate(qL, axis=0)
    os.makedirs('data/book', exist_ok=True)
    np.save('data/book/{}_{}_qB.npy'.format(args.dataset, args.num_bits), qB)
    np.save('data/book/{}_{}_qL.npy'.format(args.dataset, args.num_bits), qL)

    rB = []
    rL = []
    for i, (images, labels) in enumerate(database_loader):
        images = t.as_tensor(images.to(args.device), dtype=t.float32, device=args.device)
        labels = t.as_tensor(labels.to(args.device), dtype=label_dtype, device=args.device)
        if args.dataset == 'imagenet':
            hash_label = t.zeros((labels.size(0), 100), dtype=t.float32, device=args.device)
            hash_label.scatter_(1, labels.unsqueeze(-1), 1)
        else:
            hash_label = labels

        with t.no_grad():
            _, _, hash_code = net(images)
        rB.append(hash_code.float().cpu().numpy())
        rL.append(hash_label.float().cpu().numpy())
    rB = np.concatenate(rB, axis=0)
    rL = np.concatenate(rL, axis=0)
    np.save('data/book/{}_{}_rB.npy'.format(args.dataset, args.num_bits), rB)
    np.save('data/book/{}_{}_rL.npy'.format(args.dataset, args.num_bits), rL)


def main(args):
    if args.train:
        # print('is trian. quit')
        train(args)
    else:
        print('is test')
        test(args)
    print('over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_lr_decrease', type=int, default=80)
    parser.add_argument('--num_bits', type=int, default=16)
    parser.add_argument('--num_classes', type=int, choices=[100, 21], default=100)
    parser.add_argument('--train', type=eval)
    parser.add_argument('--dataset', type=str, default='nuswide')
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.root = os.path.join(args.root, args.dataset)
    main(args)
