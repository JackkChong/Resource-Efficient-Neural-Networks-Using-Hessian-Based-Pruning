from utils.model_utils import CIFARResNet, CIFARResNet_HAP


# =====================================================
# For network helper functions
# =====================================================
def get_network(network, depth, dataset, widening_factor, state_dict=None, from_HAP=False):
    if network == 'wideresnet' and from_HAP is False:
        return wideresnet(depth=depth, dataset=dataset, widening_factor=widening_factor)
    
    elif network == 'resnet' and from_HAP is False:
        return resnet(depth=depth, dataset=dataset)
    
    elif network == 'resnet' and from_HAP is True:
        return resnet_HAP(depth=depth, dataset=dataset, state_dict=state_dict)
    
    elif network == 'wideresnet' and from_HAP is True:
        return wideresnet_HAP(depth=depth, dataset=dataset, state_dict=state_dict)
    
    else:
        raise NotImplementedError


def wideresnet(depth, dataset, widening_factor):
    assert (depth - 4) % 6 == 0, 'Depth must be = 6n + 4, got %d' % depth

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError
    
    layers = [(depth - 4) // 6] * 3
    channels_per_layers = [16 * widening_factor, 32 * widening_factor, 64 * widening_factor]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    init_block_channels = 16

    model = CIFARResNet(channels=channels, init_block_channels=init_block_channels, num_classes=num_classes)

    return model


def resnet(depth, dataset):
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    layers = [(depth - 2) // 6] * 3
    channels_per_layers = [16, 32, 64]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    init_block_channels = 16

    model = CIFARResNet(channels=channels, init_block_channels=init_block_channels, num_classes=num_classes)

    return model


def wideresnet_HAP(depth, dataset, state_dict):
    assert (depth - 4) % 6 == 0, 'Depth must be = 6n + 4, got %d' % depth

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    stages = 3
    units = (depth - 4) // 6

    model = CIFARResNet_HAP(state_dict, stages, units, num_classes=num_classes)

    return model


def resnet_HAP(depth, dataset, state_dict):
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    stages = 3
    units = (depth - 2) // 6

    model = CIFARResNet_HAP(state_dict, stages, units, num_classes=num_classes)

    return model


