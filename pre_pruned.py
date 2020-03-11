import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from se_resnet.senet import SeResNet101x
from se_resnet.se_module import SELayer

from copy import deepcopy
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch SE Net CIFAR training')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='training dataset (default: cifar100)')
parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--percent', default=0.8, type=float, metavar='PATH',
                    help='path to latest checkpoint (default: 0.8)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes=100
if args.dataset == 'cifar100':
    num_classes=100
elif args.dataset == 'cifar10':
    num_classes=10

model = SeResNet101x(num_classes)
model = model.to(device)

if args.pretrain:
    if os.path.isfile(args.pretrain):
        print("=> loading checkpoint '{}'".format(args.pretrain))
        checkpoint = torch.load(args.pretrain)
        # args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec: {:f}"
              .format(args.pretrain, checkpoint['epoch'], best_prec))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrain))

def obtain_prune_idx(path):
    lines = []
    with open(path, 'r') as f:
        file = f.readlines()
        for line in file:
            lines.append(line)
            
    idx = 0
    prune_idx = []
    for line in lines:
        if "):" in line:
            idx  += 1
        if "bn1" in line or "bn2" in line: # 需要剪枝的层
            #print(idx, line)
            prune_idx.append(idx)
            
    return prune_idx

model_name = "seresnet.txt"
print(model, file=open(model_name, 'w'))
prune_idx = obtain_prune_idx(model_name)

def sort_bn(model, prune_idx):
    size_list = [m.weight.data.shape[0] for idx, m in enumerate(model.modules()) if idx in prune_idx]
    # bn_layer = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_prune_layers = [m for idx, m in enumerate(model.modules()) if idx in prune_idx]
    bn_weights = torch.zeros(sum(size_list))

    index = 0
    for module, size in zip(bn_prune_layers, size_list):
        bn_weights[index:(index + size)] = module.weight.data.abs().clone()
        index += size
    sorted_bn = torch.sort(bn_weights)[0]
    
    return sorted_bn

sorted_bn = sort_bn(model, prune_idx)


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask

def prune_and_eval(model, sorted_bn, prune_idx, percent=0.8, dataset='cifar100'):
    model_copy = deepcopy(model)
    thre_index = int(len(sorted_bn) * percent)
    thre = sorted_bn[thre_index]

    print('the required prune percent is', percent)
    print(f'Gamma value that less than {thre:.6f} are set to zero!')

    remain_num = 0
    
    # bn_layer = [m for m in model_copy.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_prune_layers = [m for idx, m in enumerate(model_copy.modules()) if idx in prune_idx]
    
    masks = []
    
    for bn_module in bn_prune_layers:

        mask = obtain_bn_mask(bn_module, thre)
        
        masks.append(mask)

        remain_num += int(mask.sum())
        
        bn_module.weight.data.mul_(mask)

    print("let's test the current model!")
    if dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=128, shuffle=False, num_workers = 2)
    elif dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=128, shuffle=False, num_workers = 2)
    else:
        raise ValueError("No valid dataset is given.")
        
    model_copy.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model_copy(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    print("\nTest Accuracy of 'pruned' model is: {}/{} ({:.1f}%)\n".format(
    correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')

    return thre, masks

prune_and_eval(model, sorted_bn, prune_idx, args.percent, args.dataset)



