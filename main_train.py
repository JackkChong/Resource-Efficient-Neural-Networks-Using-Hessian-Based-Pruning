import os
import torch
import argparse
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torchinfo import summary
from torch.cuda.amp import GradScaler
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from utils.common_utils import CosineLRScheduler, compute_model_param_flops, LabelSmoothing

# Globals
BEST_ACC = 0
AFFINE = True


def train(epoch, net, lr_scheduler, optimizer, criterion, trainloader, train_FP16):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)

    desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

    scaler = GradScaler(enabled=train_FP16)

    for batch_idx, (inputs, targets) in prog_bar:
        # Get data to CUDA if possible
        inputs, targets = inputs.cuda(), targets.cuda()

        # forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_FP16):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # update weights and biases
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    print(f'Train Loss: {train_loss / total}')
    print(f'Train Acc: {np.around(correct / total * 100, 2)}')


def test(epoch, net, lr_scheduler, optimizer, criterion, testloader, test_FP16, log_directory):
    global BEST_ACC
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in prog_bar:
            # Get data to CUDA if possible
            inputs, targets = inputs.cuda(), targets.cuda()

            # forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=test_FP16):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                lr_scheduler.get_lr(optimizer), test_loss / (batch_idx + 1), 100. * correct / total,
                correct, total))
            prog_bar.set_description(desc, refresh=True)

        print(f'Test Loss: {test_loss / total}')
        print(f'Test Acc: {np.around(correct / total * 100, 2)}')

    # save checkpoint
    acc = 100. * correct / total
    if acc > BEST_ACC:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
            'loss': loss,
        }

        torch.save(state, log_directory)
        BEST_ACC = acc


def main(args):
    global BEST_ACC

    cudnn.benchmark = True

    # Create directory to store results
    if os.path.isdir("results"):
        print("Results directory exists!\n")
    else:
        print("Creating results directory!\n")
        os.mkdir("results")

    # Initialize model architecture
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, widening_factor=args.widening_factor)
    summary(net, (1, 3, 32, 32), col_names=['input_size', 'output_size', 'num_params'])
    net = net.cuda()

    # Initialize data loaders
    trainloader, testloader = get_dataloader(dataset=args.dataset, train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    # Initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    # Initialize learning rate scheduler
    lr_scheduler = CosineLRScheduler(epochs=args.epochs, start_lr=args.learning_rate)

    # Initialize criterion
    criterion = LabelSmoothing(args.smoothing)

    # Calculate FLOPS
    if args.dataset == 'cifar10' or 'cifar100':
         _ = compute_model_param_flops(net, 32)
    else:
        raise NotImplementedError

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print('==> Resuming from checkpoint...\n')
        checkpoint = torch.load(f'{args.resume}', map_location=args.device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        BEST_ACC = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('==> Loaded %s trained on %s\n' % (args.network, args.dataset))
        print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%\n' % (start_epoch, BEST_ACC))

    # Iterate over epochs
    for epoch in range(start_epoch, args.epochs):
        train(epoch, net, lr_scheduler, optimizer, criterion, trainloader, args.train_FP16)
        test(epoch, net, lr_scheduler, optimizer, criterion, testloader, args.test_FP16, args.log_directory)

    # Print best accuracy
    print("Best accuracy is %.2f%%" % (BEST_ACC))


if __name__ == "__main__":
    # Fetch args
    parser = argparse.ArgumentParser(description="Train a model from scratch.")

    parser.add_argument('--dataset', default="cifar10", type=str, choices=["cifar10", "cifar100"])
    parser.add_argument('--network', default="resnet", type=str, choices=["resnet", "wideresnet"])
    parser.add_argument('--depth', default=32, type=int)
    parser.add_argument('--widening_factor', default=1, type=int)

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--nesterov', default=True, type=bool)
    parser.add_argument('--smoothing', default=0.0, type=float)

    parser.add_argument('--train_FP16', default=False, type=bool)
    parser.add_argument('--test_FP16', default=False, type=bool)

    parser.add_argument('--log_directory', default="results/resnet_32_best_1.pth.tar", type=str)
    parser.add_argument('--resume', '-r', default=None, type=str)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    
    args = parser.parse_args()

    main(args)
