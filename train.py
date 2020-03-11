import os
import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# from se_resnet.resnet import SeResNet101
from se_resnet.senet import AlphaSeResNet101, SeResNet101x

from apex import amp

parser = argparse.ArgumentParser(description='PyTorch SE Net CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# resume
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# optimizers
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num_workers',  default=2, type=int,
                    metavar='NW', help='num of workers (default: 2)')
# save
parser.add_argument('--save', default='./weights', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

args = parser.parse_args()

mixed_precision = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

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

num_classes = 10

if args.dataset == 'cifar10':
    num_classes = 10
    trainset = datasets.CIFAR10(root='./data.cifar10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testset = datasets.CIFAR10(root='./data.cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)
elif args.dataset == 'cifar100':
    num_classes = 100
    trainset = datasets.CIFAR100(root='./data.cifar100', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testset = datasets.CIFAR100(root='./data.cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)

# model = SeResNet101(num_classes)
# model = AlphaSeResNet101(num_classes)
model = SeResNet101x(num_classes)
print(model, file=open("SeResNet101x.txt", 'w'))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# criterion = F.cross_entropy()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if mixed_precision:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0 or batch_idx+1 == len(train_loader):
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\t lr:{} Loss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), args.lr, loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # test_loss += nn.CrossEntropyLoss(output, target, reduction='sum').item()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: learning rate : {} Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(args.lr, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, epoch, filepath):
    torch.save(state, os.path.join(filepath, 'model_{}.pth'.format(epoch)))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'model_{}.pth'.format(epoch)), os.path.join(filepath, 'model_best.pth'))

best_prec = 0.
for epoch in range(args.start_epoch, args.epochs):
    # lr decay !
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec = test()
    
    is_best = prec > best_prec
    if prec > best_prec:
        best_epoch = epoch
        best_prec = prec
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec': best_prec,
        'best_epoch': best_epoch,
        'optimizer': optimizer.state_dict(),
        'learning_rate': args.lr,
    }, is_best, epoch, filepath=args.save)

print("Best epoch: {} accuracy: {}".format(best_epoch, best_prec))




