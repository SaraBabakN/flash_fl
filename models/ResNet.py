import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, track, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=track)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, track, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=track)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=track)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, init_weight=True, cifar=True):
        track = True
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=track)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, track=track)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, track=track)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, track=track)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, track=track)
        if cifar:
            self.feature = nn.AvgPool2d(4, stride=1)
        else:
            self.feature = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if init_weight:
            self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, track=track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.feature(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes, cifar=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar=cifar)


def ResNet34(num_classes, cifar=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cifar=cifar)


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152(num_classes, cifar=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, cifar=cifar)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        print(p.size())
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   --                        --
├─Conv2d: 1-1                            [16, 64, 32, 32]          1,728
├─BatchNorm2d: 1-2                       [16, 64, 32, 32]          128
├─Sequential: 1-3                        [16, 64, 32, 32]          --
│    └─BasicBlock: 2-1                   [16, 64, 32, 32]          --
│    │    └─Conv2d: 3-1                  [16, 64, 32, 32]          36,864
│    │    └─BatchNorm2d: 3-2             [16, 64, 32, 32]          128
│    │    └─Conv2d: 3-3                  [16, 64, 32, 32]          36,864
│    │    └─BatchNorm2d: 3-4             [16, 64, 32, 32]          128
│    │    └─Sequential: 3-5              [16, 64, 32, 32]          --
│    └─BasicBlock: 2-2                   [16, 64, 32, 32]          --
│    │    └─Conv2d: 3-6                  [16, 64, 32, 32]          36,864
│    │    └─BatchNorm2d: 3-7             [16, 64, 32, 32]          128
│    │    └─Conv2d: 3-8                  [16, 64, 32, 32]          36,864
│    │    └─BatchNorm2d: 3-9             [16, 64, 32, 32]          128
│    │    └─Sequential: 3-10             [16, 64, 32, 32]          --
├─Sequential: 1-4                        [16, 128, 16, 16]         --
│    └─BasicBlock: 2-3                   [16, 128, 16, 16]         --
│    │    └─Conv2d: 3-11                 [16, 128, 16, 16]         73,728
│    │    └─BatchNorm2d: 3-12            [16, 128, 16, 16]         256
│    │    └─Conv2d: 3-13                 [16, 128, 16, 16]         147,456
│    │    └─BatchNorm2d: 3-14            [16, 128, 16, 16]         256
│    │    └─Sequential: 3-15             [16, 128, 16, 16]         8,448
│    └─BasicBlock: 2-4                   [16, 128, 16, 16]         --
│    │    └─Conv2d: 3-16                 [16, 128, 16, 16]         147,456
│    │    └─BatchNorm2d: 3-17            [16, 128, 16, 16]         256
│    │    └─Conv2d: 3-18                 [16, 128, 16, 16]         147,456
│    │    └─BatchNorm2d: 3-19            [16, 128, 16, 16]         256
│    │    └─Sequential: 3-20             [16, 128, 16, 16]         --
├─Sequential: 1-5                        [16, 256, 8, 8]           --
│    └─BasicBlock: 2-5                   [16, 256, 8, 8]           --
│    │    └─Conv2d: 3-21                 [16, 256, 8, 8]           294,912
│    │    └─BatchNorm2d: 3-22            [16, 256, 8, 8]           512
│    │    └─Conv2d: 3-23                 [16, 256, 8, 8]           589,824
│    │    └─BatchNorm2d: 3-24            [16, 256, 8, 8]           512
│    │    └─Sequential: 3-25             [16, 256, 8, 8]           33,280
│    └─BasicBlock: 2-6                   [16, 256, 8, 8]           --
│    │    └─Conv2d: 3-26                 [16, 256, 8, 8]           589,824
│    │    └─BatchNorm2d: 3-27            [16, 256, 8, 8]           512
│    │    └─Conv2d: 3-28                 [16, 256, 8, 8]           589,824
│    │    └─BatchNorm2d: 3-29            [16, 256, 8, 8]           512
│    │    └─Sequential: 3-30             [16, 256, 8, 8]           --
├─Sequential: 1-6                        [16, 512, 4, 4]           --
│    └─BasicBlock: 2-7                   [16, 512, 4, 4]           --
│    │    └─Conv2d: 3-31                 [16, 512, 4, 4]           1,179,648
│    │    └─BatchNorm2d: 3-32            [16, 512, 4, 4]           1,024
│    │    └─Conv2d: 3-33                 [16, 512, 4, 4]           2,359,296
│    │    └─BatchNorm2d: 3-34            [16, 512, 4, 4]           1,024
│    │    └─Sequential: 3-35             [16, 512, 4, 4]           132,096
│    └─BasicBlock: 2-8                   [16, 512, 4, 4]           --
│    │    └─Conv2d: 3-36                 [16, 512, 4, 4]           2,359,296
│    │    └─BatchNorm2d: 3-37            [16, 512, 4, 4]           1,024
│    │    └─Conv2d: 3-38                 [16, 512, 4, 4]           2,359,296
│    │    └─BatchNorm2d: 3-39            [16, 512, 4, 4]           1,024
│    │    └─Sequential: 3-40             [16, 512, 4, 4]           --
├─Linear: 1-7                            [16, 10]                  5,130
==========================================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
Total mult-adds (G): 8.89
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 157.29
Params size (MB): 44.70
Estimated Total Size (MB): 202.18
==========================================================================================
"""
