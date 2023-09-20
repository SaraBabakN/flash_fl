import os
import PIL
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, Dataset

from data_prepration.common import create_lda_partitions


def get_dataset(args):
    """Creates augmented train, validation, and test data loaders."""
    if args.dataset == 'cifar10':
        return get_cifar10(args)
    elif args.dataset == 'cifar100':
        return get_cifar100(args)
    elif args.dataset == 'mnist':
        return get_mnist(args)
    elif args.dataset == 'tinyimagenet':
        return get_tiny(args)
    else:
        raise NotImplementedError


def get_cifar10(args):
    data_dir = args.path
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))   # #
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    train_user_groups, test_user_groups = get_lda_distribution(args, train_dataset, test_dataset)
    return train_dataset, test_dataset, train_user_groups, test_user_groups


def get_mnist(args):
    data_dir = args.path
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)
    train_user_groups, test_user_groups = get_lda_distribution(args, train_dataset, test_dataset)
    return train_dataset, test_dataset, train_user_groups, test_user_groups


def get_cifar100(args):
    data_dir = args.path
    normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    train_user_groups, test_user_groups = get_lda_distribution(args, train_dataset, test_dataset)
    return train_dataset, test_dataset, train_user_groups, test_user_groups


def get_lda_distribution(args, train_dataset, test_dataset):
    train_user_groups, dirichlet_dist = create_lda_partitions(dataset=np.array(train_dataset.targets),
                                                              dirichlet_dist=None,
                                                              num_partitions=args.num_users,
                                                              concentration=args.alpha,
                                                              accept_imbalanced=False,)
    train_user_groups = [train_user_groups[i][0].tolist() for i in range(len(train_user_groups))]

    concentration = args.alpha
    test_user_groups, _ = create_lda_partitions(dataset=np.array(test_dataset.targets),
                                                dirichlet_dist=None,
                                                num_partitions=args.num_users,
                                                concentration=concentration,
                                                accept_imbalanced=False,)
    test_user_groups = [test_user_groups[i][0].tolist() for i in range(len(test_user_groups))]

    return train_user_groups, test_user_groups


def get_partitioned_data(args):
    train_dataset, test_dataset, train_user_groups, test_user_groups = get_dataset(args)
    all_test_data = DataLoader(test_dataset, batch_size=128, shuffle=False)
    clients_train_data = {}
    clients_test_data = {}
    clients_weights = {}
    for i in range(args.num_users):
        train_dataset_i = Subset(train_dataset, train_user_groups[i])
        train_loader_i = DataLoader(train_dataset_i, batch_size=args.local_bs, shuffle=True)
        test_dataset_i = Subset(test_dataset, test_user_groups[i])
        test_loader_i = DataLoader(test_dataset_i, batch_size=args.local_bs, shuffle=False)
        clients_train_data[i] = train_loader_i
        clients_test_data[i] = test_loader_i
        clients_weights[i] = len(train_user_groups[i])
    return all_test_data, clients_train_data, clients_test_data, clients_weights


class CustomImageDataset(Dataset):
    def __init__(self, x, y):
        self.targets = y
        self.x = x

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.x[idx], self.targets[idx]


def get_tiny(args):

    def parse_classes(file):
        classes = []
        filenames = []
        with open(file) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for x in range(len(lines)):
            tokens = lines[x].split()
            classes.append(tokens[1])
            filenames.append(tokens[0])
        return filenames, classes

    class TinyImageNetDataset(torch.utils.data.Dataset):
        """Dataset wrapping images and ground truths."""
        def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
            self.img_path = img_path
            self.transform = transform
            self.gt_path = gt_path
            self.class_to_idx = class_to_idx
            self.classidx = []
            self.imgs, self.classnames = parse_classes(gt_path)
            for classname in self.classnames:
                self.classidx.append(self.class_to_idx[classname])
            self.targets = self.classidx

        def __getitem__(self, index):
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

        def __len__(self):
            return len(self.imgs)

    data_path = args.path
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'tiny-imagenet-200', 'train'),
        transform=transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    )

    test_dataset = TinyImageNetDataset(
        img_path=os.path.join(data_path, 'tiny-imagenet-200', 'val', 'images'),
        gt_path=os.path.join(data_path, 'tiny-imagenet-200', 'val', 'val_annotations.txt'),
        class_to_idx=train_dataset.class_to_idx.copy(),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    )
    train_user_groups, test_user_groups = get_lda_distribution(args, train_dataset, test_dataset)
    return train_dataset, test_dataset, train_user_groups, test_user_groups
