import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import torch.utils.model_zoo as model_zoo
import numpy as np


device = "cpu"
## device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False) # ELN: modified as in VGG_FB
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False) # ELN: modified
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_FB(nn.Module): # ELN: modified

    def __init__(self, block, layers, layer_sizes, num_classes=1000, zero_init_residual=False):  # ELN: adding layer_sizes as in VGG_FB
        super(ResNet_FB, self).__init__() # ELN: modified
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False) # ELN: modified as in VGG_FB
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # ELN: modified from Feedback-CNN
        self.layer_sizes = layer_sizes
        self.z = {}
        i=0
        for _ in self.layer_sizes:
            self.z[i] = torch.ones(self.layer_sizes[i]).to(device)
            i += 1

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reset(self): # ELN: adding from Feedback-CNN
        for i in self.z.keys():
            self.z[i] = torch.ones(self.layer_sizes[i]).to(device)

    # ELN: currently copied/adapted from vgg_fb.py
    def forward(self, x, i_layer=10000):

        print("FORWARD! (ResNet_FB)")

        layerlist = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool, self.fc]

        self.input = []
        self.output = []

        #i=0
        for layer in layerlist:
        #    if i>i_layer: break
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            if layer==self.fc: # reshape before fc layer
                x = x.view(x.size(0), -1)
            x = layer(x)
            self.output.append(x)
        #    i += 1

        return x

    # ELN: currently this is just copied from vgg_fb.py    
    def backward(self, g, i_out=None, i_in=0):
        if i_out is None:
            i_out = len(self.output)
        for i, output in reversed(list(enumerate(self.output))):
            if i>i_out-1:
                continue
            if i==i_out-1:
                output.backward(g)
            if i<i_out-1:
                output.backward(self.input[i+1].grad.data)
            if i in self.z:
                alpha = self.input[i].grad
                self.z[i] = (alpha > 0).float()
                self.input[i].grad = self.z[i] * alpha
            if i not in self.z:
                print('i is not in self.z ... figure out what is going on')
                exit()
            if i==i_in:
                break
        return self.input[i_in].grad


def resnet18_fb(pretrained=False, **kwargs): # ELN: modified
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_FB(BasicBlock, [2, 2, 2, 2], **kwargs) # ELN: modified
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
