import torch
import argparse
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchinfo import summary
from utils.network_utils import get_network
from utils.data_utils import get_dataloader, get_hessianloader
from utils.common_utils import CosineLRScheduler, count_parameters, compute_ratio, compute_model_param_flops
from prune_utils.pruner import HessianPruner

# Globals
BEST_ACC = 0
AFFINE = True


def main(args):
    global BEST_ACC

    cudnn.benchmark = True

    # Initialize model architecture
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset,
                      widening_factor=args.widening_factor)
    summary(net, (1, 3, 32, 32), col_names=['input_size', 'output_size', 'num_params'])

    # Loading pre-trained model from checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
    print("Acc: %.2f%%, Epoch: %d, Loss: %.4f\n" % (checkpoint['acc'], checkpoint['epoch'], checkpoint['loss']))
    state_dict = checkpoint['net']
    net.load_state_dict(state_dict)
    total_parameters = count_parameters(net)
    net = net.cuda()

    # Initialize data loaders
    trainloader, testloader = get_dataloader(dataset=args.dataset, train_batch_size=args.batch_size,
                                             test_batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    # Initialize hessian loader
    hess_data = []
    hessianloader = get_hessianloader(args.dataset, args.hessian_batch_size)
    for data, label in hessianloader:
        hess_data = (data, label)

    # Calculate FLOPS before pruning
    total_flops = 0
    if args.dataset == 'cifar10' or 'cifar100':
        total_flops = compute_model_param_flops(net, 32)
    else:
        raise NotImplementedError

    # Initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    # Initialize learning rate scheduler
    lr_scheduler = CosineLRScheduler(epochs=args.epochs, start_lr=args.learning_rate)

    # Initialize criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize pruner
    pruner = HessianPruner(net, optimizer, lr_scheduler, args.prune_ratio, args.prune_ratio_limit,
                           args.fix_layers, args.hessian_mode, args.use_decompose)

    # Conduct pruning
    _ = pruner.make_pruned_model(hess_data, criterion, args.n_v, args.trace_directory, args.network, args.trace_FP16)

    # Track the compression ratio
    compression_ratio, pruned_numel = compute_ratio(pruner.model, total_parameters)
    print("Compression ratio: %.2f%%(%d/%d)\n" % (compression_ratio, pruned_numel, total_parameters))

    pruned_flops = 0
    # Calculate FLOPS after pruning
    if args.dataset == 'cifar10' or 'cifar100':
        pruned_flops = compute_model_param_flops(pruner.model, 32)
    elif args.dataset == 'imagenet':
        pruned_flops = compute_model_param_flops(pruner.model, 224)

    print("Remained flops: %.2f%%" % (pruned_flops / total_flops * 100))

    # Start finetuning
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        pruner.finetune_model(epoch, criterion, trainloader, args.finetune_FP16)
        BEST_ACC = pruner.test_model(epoch, criterion, testloader, args.test_FP16, args.log_directory, BEST_ACC)

    # Print model size and best accuracy
    print("Best accuracy is %.2f%%" % (BEST_ACC))


if __name__ == "__main__":
    # Fetch args
    parser = argparse.ArgumentParser(description="Prune and finetune a pre-trained model.")

    parser.add_argument('--dataset', default="cifar10", type=str, choices=["cifar10", "cifar100"])
    parser.add_argument('--network', default="resnet", type=str, choices=["resnet", "wideresnet"])
    parser.add_argument('--depth', default=32, type=int)
    parser.add_argument('--widening_factor', default=1, type=int)

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--nesterov', default=True, type=bool)

    parser.add_argument('--finetune_FP16', default=False, type=bool)
    parser.add_argument('--test_FP16', default=False, type=bool)

    parser.add_argument('--n_v', default=300, type=int)
    parser.add_argument('--hessian_batch_size', default=512, type=int)
    parser.add_argument('--prune_ratio', default=0.75214, type=float)
    parser.add_argument('--prune_ratio_limit', default=0.95, type=float)
    parser.add_argument('--fix_layers', default=0, type=int)
    parser.add_argument('--hessian_mode', default="trace", type=str, choices=["trace"])
    parser.add_argument('--use_decompose', default=False, type=bool)
    parser.add_argument('--trace_FP16', default=True, type=bool)

    parser.add_argument('--load_checkpoint', default="results/resnet_32_best_1.pth.tar", type=str)
    parser.add_argument('--log_directory', default="results/resnet_32_pruned_best_1.pth.tar", type=str)
    parser.add_argument('--trace_directory', default="results/resnet_32_pruned.npy", type=str)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)

    args = parser.parse_args()

    main(args)
