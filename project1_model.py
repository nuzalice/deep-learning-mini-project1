import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, kernel_size, skip_kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size,
                               stride=1, padding=int((kernel_size-1)/2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          skip_kernel_size, stride=stride, padding=int((skip_kernel_size-1)/2), bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, N, num_blocks, C_1, conv_kernel_size, skip_kernel_size, avg_pool_size, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = C_1
        self.avg_pool_size = avg_pool_size

        self.conv1 = nn.Conv2d(3, C_1, conv_kernel_size[0],
                               stride=1, padding=int((conv_kernel_size[0]-1)/2), bias=False)
        self.bn1 = nn.BatchNorm2d(C_1)

        res_layers = []
        
        for i in range(N):
          if i == 0:
            s = 1
          else:
            s = 2
          res_layers.append(self._make_layer(block, (2**i)*C_1, num_blocks[i], conv_kernel_size[i], skip_kernel_size[i], stride=s))

        self.layer = nn.Sequential(*res_layers)

        ps = 2**(np.ceil(np.log2(avg_pool_size)))


        self.linear = nn.Linear(int(C_1*32*32/(2**(N-1))/(ps**2)), num_classes)

    def _make_layer(self, block, planes, num_blocks, conv_kernel_size, skip_kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, conv_kernel_size, skip_kernel_size, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = F.avg_pool2d(out, self.avg_pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model(N=3, num_blocks=[2,2,2], C_1=85, conv_kernel_size=[3,3,3], skip_kernel_size=[1,1,1], avg_pool_size=8):
    return ResNet(BasicBlock, N, num_blocks, C_1, conv_kernel_size, skip_kernel_size, avg_pool_size, num_classes=10)
    


