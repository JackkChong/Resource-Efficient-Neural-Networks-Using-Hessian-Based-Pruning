import torch
import torchvision
from torchvision import transforms


# =====================================================
# For data loaders and data transforms
# =====================================================
def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=1, root='data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader


def get_transforms(dataset):
    transform_train = None
    transform_test = None

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset

    return transform_train, transform_test


def get_hessianloader(dataset, hessian_batch_size, num_workers=1, root='data'):
    transform_hessian = get_transforms_hessian(dataset)
    hessianset = None

    if dataset == 'cifar10':
        hessianset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_hessian)

    elif dataset == 'cifar100':
        hessianset = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform_hessian)

    assert hessianset is not None, 'Error, no dataset %s' % dataset

    hessian_loader = torch.utils.data.DataLoader(hessianset, batch_size=hessian_batch_size, shuffle=False,
                                                 num_workers=num_workers)

    return hessian_loader


def get_transforms_hessian(dataset):
    transform_hessian = None

    if dataset == 'cifar10':
        transform_hessian = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    if dataset == 'cifar100':
        transform_hessian = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    assert transform_hessian is not None, 'Error, no dataset %s' % dataset

    return transform_hessian
