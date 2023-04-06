import re
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torchinfo import summary
from utils.data_utils import get_dataloader
from utils.common_utils import CosineLRScheduler
from utils.network_utils import get_network
from quant_utils.quant_modules import freeze_model, unfreeze_model, quantize_resnet, quantize_wideresnet

# Globals
BEST_ACC = 0
AFFINE = True


def train(epoch, net, lr_scheduler, optimizer, criterion, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)

    desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

    for batch_idx, (inputs, targets) in prog_bar:
        # Get data to CUDA if possible
        inputs, targets = inputs.cuda(), targets.cuda()

        # forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # update weights and biases
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    print(f'Train Loss: {train_loss / total}')
    print(f'Train Acc: {np.around(correct / total * 100, 2)}')


def test(net, lr_scheduler, optimizer, criterion, testloader, save_checkpoint):
    global BEST_ACC
    freeze_model(net)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    desc = ('[LR=%.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            # Get data to CUDA if possible
            inputs, targets = inputs.cuda(), targets.cuda()

            # forward pass
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

    # save quantized checkpoint
    acc = 100. * correct / total
    if acc > BEST_ACC:
        print('Saving..')
        torch.save({'convbn_scaling_factor': {k: v for k, v in net.state_dict().items() if 'convbn_scaling_factor' in k},
                    'fc_scaling_factor': {k: v for k, v in net.state_dict().items() if 'fc_scaling_factor' in k},
                    'weight_integer': {k: v for k, v in net.state_dict().items() if 'weight_integer' in k},
                    'bias_integer': {k: v for k, v in net.state_dict().items() if 'bias_integer' in k},
                    'act_scaling_factor': {k: v for k, v in net.state_dict().items() if 'act_scaling_factor' in k},
                    }, save_checkpoint)
        BEST_ACC = acc

    unfreeze_model(net)


def main(args):
    global BEST_ACC

    cudnn.benchmark = True

    # Loading pre-trained model from checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
    print("Acc: %.2f%%, Epoch: %d, Loss: %.4f\n" % (checkpoint['acc'], checkpoint['epoch'], checkpoint['loss']))
    state_dict = checkpoint['net']

    # Initialize model architecture
    model = get_network(network=args.network, depth=args.depth, dataset=args.dataset,
                      widening_factor=args.widening_factor, state_dict=state_dict, from_HAP=args.from_HAP)
    model.load_state_dict(state_dict)
    model = model.cuda()
    summary(model, (1, 3, 32, 32), col_names=['input_size', 'output_size', 'num_params'])

    # Initialize data loaders
    trainloader, testloader = get_dataloader(dataset=args.dataset, train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    # Generate quantized model
    if args.network == 'wideresnet':
        qmodel = quantize_wideresnet(model, args.depth)
    else:
        qmodel = quantize_resnet(model, args.depth)

    # Creating bit configuration
    bit_config = {}
    bit_config['quant_input'] = 8
    bit_config['quant_init_block_convbn'] = 8
    bit_config['quant_act_int32'] = 16
    bit_config['quant_act_output'] = 8
    bit_config['quant_output'] = 8

    num_stages = 3
    if args.network == 'resnet':
        num_units = (args.depth - 2) // 6
    else:
        num_units = (args.depth - 4) // 6

    for stage in range(num_stages):
        for  unit in range(num_units):
            if args.quant_scheme == 'uniform8':
                bit_config['stage{}.unit{}.quant_act'.format(stage + 1, unit + 1)] = 8
                bit_config['stage{}.unit{}.quant_convbn1'.format(stage + 1, unit + 1)] = 8
                bit_config['stage{}.unit{}.quant_act1'.format(stage + 1, unit + 1)] = 8
                bit_config['stage{}.unit{}.quant_convbn2'.format(stage + 1, unit + 1)] = 8
            
            elif args.quant_scheme == 'uniform4':
                bit_config['stage{}.unit{}.quant_act'.format(stage + 1, unit + 1)] = 4
                bit_config['stage{}.unit{}.quant_convbn1'.format(stage + 1, unit + 1)] = 4
                bit_config['stage{}.unit{}.quant_act1'.format(stage + 1, unit + 1)] = 4
                bit_config['stage{}.unit{}.quant_convbn2'.format(stage + 1, unit + 1)] = 4
            
            else:
                raise NotImplementedError
            
            bit_config['stage{}.unit{}.quant_act_int32'.format(stage + 1, unit + 1)] = 16
    
    name_counter = 0
    for name, m in qmodel.named_modules():
        if name in bit_config.keys():
            name_counter += 1

            # Set attributes based on quantization settings
            if re.search('(quant_act$)|(quant_act1$)', name):
                setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', args.bias_bit)
            setattr(m, 'quantize_bias', (args.bias_bit != 0))
            setattr(m, 'per_channel', args.channel_wise)
            setattr(m, 'act_percentile', args.act_percentile)
            setattr(m, 'act_range_momentum', args.act_range_momentum)
            setattr(m, 'weight_percentile', args.weight_percentile)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', args.fix_bn)
            setattr(m, 'fix_BN_threshold', args.fix_bn_threshold)
            setattr(m, 'training_BN_mode', args.fix_bn)
            setattr(m, 'fixed_point_quantization', args.fixed_point_quantize)

            # Set bidwidth for each module
            bitwidth = bit_config[name]
            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
            else:
                setattr(m, 'weight_bit', bitwidth)

    print("Matched all modules defined in bit_config: {}".format(len(bit_config.keys()) == name_counter))
    qmodel = qmodel.cuda()
    summary(qmodel, (1, 3, 32, 32), depth=1, col_names=['input_size', 'output_size', 'num_params'])

    # Initialize optimizer
    optimizer = torch.optim.SGD(qmodel.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # Initialize learning rate scheduler
    lr_scheduler = CosineLRScheduler(epochs=args.epochs, start_lr=args.learning_rate)

    # Initialize criterion
    criterion = nn.CrossEntropyLoss()

    # Iterate over epochs
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        train(epoch, qmodel, lr_scheduler, optimizer, criterion, trainloader)
        test(qmodel, lr_scheduler, optimizer, criterion, testloader, args.save_checkpoint)
    
    # Print model size and best accuracy
    print("Best accuracy is %.2f%%" % (BEST_ACC))


if __name__ == "__main__":
    # Fetch args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="cifar10", type=str)
    parser.add_argument('--network', default="resnet", type=str)
    parser.add_argument('--from_HAP', default=True, type=bool)
    parser.add_argument('--depth', default=32, type=int)
    parser.add_argument('--widening_factor', default=1, type=int)

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--quant_scheme', default="uniform8", type=str)
    parser.add_argument('--bias_bit', default=32, type=int)
    parser.add_argument('--channel_wise', default=True, type=bool)
    parser.add_argument('--act_percentile', default=0, type=float)
    parser.add_argument('--act_range_momentum', default=0.99, type=float)
    parser.add_argument('--weight_percentile', default=0.0, type=float)
    parser.add_argument('--fix_bn', default=True, type=bool)
    parser.add_argument('--fix_bn_threshold', default=None)
    parser.add_argument('--fixed_point_quantize', default=False, type=bool)

    parser.add_argument('--save_checkpoint', default="results/resnet_32_quantized_best_1.pth.tar", type=str)
    parser.add_argument('--load_checkpoint', default="results/resnet_32_pruned_best_1.pth.tar", type=str)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)

    args = parser.parse_args()

    main(args)
