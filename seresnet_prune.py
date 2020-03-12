import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from se_resnet.senet import SeResNet101x
from se_resnet.se_module import SELayer
from se_resnet.new_model import pruned_model

from copy import deepcopy
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch SE Net CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--percent', default=0.6, type=float, metavar='PATH',
                    help='path to latest checkpoint (default: 0.6)')
parser.add_argument('--save', default='./pruned_weights', type=str, metavar='PATH',
                    help='path to save (default: none)')
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
else:
     print("=> please load origional model")


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


def obtain_bn_threshold(model, sorted_bn, percentage):
    thre_index = int(len(sorted_bn) * percentage)
    thre = sorted_bn[thre_index]
    
    return thre

threshold = obtain_bn_threshold(model, sorted_bn, args.percent)
print(threshold)

def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask

def obtain_filters_mask(model, prune_idx, thre):
    pruned = 0
    total = 0
    num_filters = []
    pruned_filters = []
    filters_mask = []
    pruned_maskers = []
    
    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.BatchNorm2d):
            if idx in prune_idx:
                mask = obtain_bn_mask(module, thre).cpu().numpy()
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                if remain == 0: # 保证至少有一个channel
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = module.weight.data.abs().max()
                    mask = obtain_bn_mask(module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
                
                pruned_filters.append(remain)
                pruned_maskers.append(mask.copy())
            else:
                mask = np.ones(module.weight.data.shape)
                remain = mask.shape[0]
            
            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())
    
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask, pruned_filters, pruned_maskers

num_filters, filters_mask, pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold)

print(pruned_filters, file=open("cfg.txt","w"))
new_model = pruned_model(num_classes, cfg = pruned_filters)
new_model = new_model.to(device)

def obtain_input_mask_idx(layer_id, prune_idx, pruned_maskers):
    mask_idx = []
    for idx, maskers in zip(prune_idx, pruned_maskers):
        if layer_id == idx:
            mask_idx = np.argwhere(maskers)[:, 0].tolist()
    
    return mask_idx

def init_weights_from_loose_model(model, new_model, prune_idx, pruned_maskers):
    old_modules = list(model.modules())
    new_modules = list(new_model.modules())
    
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            
            if layer_id in prune_idx:
                out_channel_idx = obtain_input_mask_idx(layer_id, prune_idx, pruned_maskers)
                
                m1.weight.data = m0.weight.data[out_channel_idx].clone()
                m1.bias.data = m0.bias.data[out_channel_idx].clone()
                m1.running_mean.data = m0.running_mean.data[out_channel_idx].clone()
                m1.running_var.data = m0.running_var.data[out_channel_idx].clone()              
            
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
        
        if isinstance(m0, nn.Conv2d):
            if layer_id+1 in prune_idx and layer_id-1 in prune_idx:
                in_channel_idx = obtain_input_mask_idx(layer_id-1, prune_idx, pruned_maskers)
                out_channel_idx = obtain_input_mask_idx(layer_id+1, prune_idx, pruned_maskers)
                
                tmp = m0.weight.data[:, in_channel_idx, :, :].clone()
                m1.weight.data = tmp[out_channel_idx, :, :, :].clone()
            elif layer_id+1 in prune_idx:
                out_channel_idx = obtain_input_mask_idx(layer_id+1, prune_idx, pruned_maskers)
                
                m1.weight.data = m0.weight.data[out_channel_idx, :, :, :]
            elif layer_id-1 in prune_idx:
                in_channel_idx = obtain_input_mask_idx(layer_id-1, prune_idx, pruned_maskers)
                
                m1.weight.data = m0.weight.data[:, in_channel_idx, :, :]
            else:
                m1.weight.data = m0.weight.data.clone()
        
        
        if isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            if layer_id == len(old_modules) - 1:
                m1.bias.data = m0.bias.data.clone()

init_weights_from_loose_model(model, new_model, prune_idx, pruned_maskers)

def model_eval(model,  dataset='cifar100'):
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
        
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    print("\nTest Accuracy of 'pruned' model is: {}/{} ({:.1f}%)\n".format(
    correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

model_eval(new_model, args.dataset)

if args.save:
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    torch.save(new_model.state_dict(), args.save +"/new_model.pth")



