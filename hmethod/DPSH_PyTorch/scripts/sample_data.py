import argparse
import os
import shutil
import torch as t


def main(args):
    basedir = os.path.abspath('./')
    dir0 = os.path.join(basedir, args.dir0)
    dir1 = os.path.join(basedir, args.dir1)
    file = os.path.join(basedir, args.txt)
    with open(file, 'r') as f:
        lines = f.readlines()
    files_from = [file.split()[0] for file in lines]

    num_file = 0
    for file in files_from:
        file_name = os.path.basename(file)
        file_dir1 = os.path.join(dir1, file_name[:9])
        os.makedirs(file_dir1, exist_ok=True)
        file_from = os.path.join(dir0, file)
        file_to = os.path.join(file_dir1, file_name)
        shutil.copy(file_from, file_to)
        num_file += 1
    print('Num:{}'.format(num_file))


def f0(args):
    files = []
    for basedir, dirs, names in os.walk(args.dir0):
        for name in names:
            if name.endswith('.JPEG'):
                files.append(name)
    with open('val_yx.txt', 'w') as f:
        f.write('\n'.join(files))
    print(len(files))


def f1(args):
    with open(args.dir0, 'r') as f:
        files0 = f.readlines()
    with open(args.dir1, 'r') as f:
        files1 = f.readlines()
    files1 = [f.split()[0] for f in files1]
    files0 = sorted(files0)
    files1 = sorted(files1)
    assert len(files0) == len(files1), 'size error'
    for i in range(len(files0)):
        if files0[i] != files1[i]:
            print('here')
            print(i, files0[i], files1[i])
            raise RuntimeError()
    print('Right??')


def get_imagenet_val(args):
    labels = [0]
    labels_name = ['None']
    with open(args.txt_label, 'r') as f:
        labels += list(map(int, f.readlines()))
    with open(args.txt_label_name, 'r') as f:
        names = f.readlines()
    labels_name += [n.split()[1] for n in names]
    os.makedirs(args.dir1, exist_ok=True)

    labels_here = set()
    with open(args.txt, 'r') as f:
        files = [f.split()[0] for f in f.readlines()]
    for file in files:
        name = os.path.basename(file)
        idx = int(name[-13:-5])
        label_name = labels_name[labels[idx]]
        labels_here.add(label_name)
        target_dir = os.path.join(args.dir1, 'val_new', label_name)
        os.makedirs(target_dir, exist_ok=True)
        src_file = os.path.join(args.dir0, file)
        target_file = os.path.join(target_dir, name)
        shutil.copy(src_file, target_file)
    print(len(labels_here))
    print(labels_here)


def transfer_imagenet_val(args):
    labels = [0]
    labels_name = ['None']
    with open(args.txt_label, 'r') as f:
        labels += list(map(int, f.readlines()))
    with open(args.txt_label_name, 'r') as f:
        names = f.readlines()
    labels_name += [n.split()[1] for n in names]

    labels_here = set()
    with open(args.txt, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        name_label = line.split()
        file = name_label[0]
        label = ' '.join(name_label[1:])
        name = os.path.basename(file)
        idx = int(name[-13:-5])
        label_name = labels_name[labels[idx]]
        labels_here.add(label_name)
        name_new = os.path.join(label_name, name)

        new_lines.append('{} {}\n'.format(name_new, label))
    print(len(labels_here))
    print(labels_here)
    with open('./new_query.txt', 'w') as f:
        f.writelines(new_lines)


def get_imagenet_train(args):
    labels_here = set()
    with open(args.txt, 'r') as f:
        files = [f.split()[0] for f in f.readlines()]
    for file in files:
        name = os.path.basename(file)
        label_name = name[:9]
        labels_here.add(label_name)
        target_dir = os.path.join(args.dir1, 'abcde', label_name)
        os.makedirs(target_dir, exist_ok=True)
        src_file = os.path.join(args.dir0, file)
        target_file = os.path.join(target_dir, name)
        shutil.copy(src_file, target_file)
    print(len(labels_here))
    print(labels_here)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir0', type=str, default='train')
    parser.add_argument('--txt', type=str, default='train.txt')
    parser.add_argument('--dir1', type=str, default='new')
    parser.add_argument('--txt_label', type=str, default='ILSVRC2012_validation_ground_truth.txt')
    parser.add_argument('--txt_label_name', type=str, default='ILSVRC2012_mapping.txt')
    parser.add_argument('--name', type=str, default='abcde')
    args = parser.parse_args()
    # main(args)
    merge_book()
