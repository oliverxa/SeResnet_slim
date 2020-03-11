import torch
import torch.nn as nn
import torch.nn.functional as F
from se_resnet.se_module import SELayer
# from se_module import SELayer

class SeBottleneck(nn.Module):
    """ SeBottleneck """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False):
        super(SeBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out

class SEResnet(nn.Module): # alphapose
    """ SEResnet with maxpooling"""

    def __init__(self,  block, num_blocks, num_classes=10):
        super(SEResnet, self).__init__()

        self.inplanes = 64
        self.block = block

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

#     def stages(self):
#         return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn0(self.conv0(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32

        # x = F.avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class SEResnet2(nn.Module):
    """ SEResnet without maxpooling"""

    def __init__(self,  block, num_blocks, num_classes=10):
        super(SEResnet2, self).__init__()

        self.inplanes = 64
        self.block = block

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.relu(self.bn0(self.conv0(x))) # 64 * h/2 * w/2
        x = self.layer1(x)  # 256 * h/2 * w/2
        x = self.layer2(x)  # 512 * h/4 * w/4
        x = self.layer3(x)  # 1024 * h/8 * w/8
        x = self.layer4(x)  # 2048 * h/16 * w/16

        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def AlphaSeResNet101(num_classes=10): # maxpooling
    return SEResnet(SeBottleneck, [3,4,23,3], num_classes=num_classes)

def SeResNet101x(num_classes=10): # without maxpooling
    return SEResnet2(SeBottleneck, [3,4,23,3], num_classes=num_classes)

def test():
    net = SeResNet101x()
    y = net(torch.randn(1,3,32,32))
    print(net, file=open("SeResNet101x.txt","w"))
    print(y.size())

if __name__ == '__main__':
    test()


