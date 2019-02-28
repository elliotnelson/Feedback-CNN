import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import numpy as np


device = "cpu"
## device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = [
   'VGG_FB', 'vgg11_fb', 'vgg11_fb_bn', 'vgg13_fb', 'vgg13_fb_bn', 'vgg16_fb', 'vgg16_fb_bn',
    'vgg19_fb_bn', 'vgg19_fb',
]


class VGG_FB(nn.Module):
    def __init__(self, features, layer_sizes, num_classes=1000):
        super(VGG_FB, self).__init__()
        classifiers = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ]
        self.layers = nn.ModuleList(features + classifiers)
        self.reshape_feature_layer_num = len(features)
        self.layer_sizes = layer_sizes
        # Add hidden gates for selective layers such as ReLU and Max-pooling
        self.z = {}
        for i, layer in list(enumerate(self.layers)):
            # if isinstance(layer, nn.ReLU):
            #     self.z[i] = torch.ones(layer_sizes[i])
            # elif isinstance(layer, nn.MaxPool2d):
            #     self.z[i] = torch.ones(layer_sizes[i])
            self.z[i] = torch.ones(self.layer_sizes[i]).to(device)
    
    def reset(self):
        for i in self.z.keys():
            self.z[i] = torch.ones(self.layer_sizes[i]).to(device)
    
    def forward(self, x, i_layer=10000): # the default i_layer just lets forward pass go all the way to last layer, recovering original method
        print("FORWARD! (VGG_FB)")
        self.input = []
        self.output = []
        for i, layer in list(enumerate(self.layers)):
            if i>i_layer: break  # ELN; end the forward pass when we arrive at layer of interest
            # detach from previous history
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            ## if i in self.z:
            ##    # multiply by the hidden gate
            ##    x = x * self.z[i]
            if i == self.reshape_feature_layer_num:
                x = x.view(x.size(0), -1)
            # compute output
            x = layer(x)  # note that layer() is element of self.layers, so, either classifier or feature (class arg) above
            # add to list of outputs
            self.output.append(x)
        return x  ## this should be self.output[i_layer] for i_layer < number of layers, assuming i starts at 0
        ## note: self.output[0] should be output of 1st layer; for i=0 ??

    # backpropagates from layer i_out to i_in; generalizes original backward(self, g)
    def backward(self, g, i_out=None, i_in=0):
        # print("BACKWARD!")
        if i_out is None:
            i_out = len(self.output)  # backward pass from output layer, by default
        for i, output in reversed(list(enumerate(self.output))):  # reversed() so that we proceed from last layer to first (i=0?)
            if i>i_out-1:
                continue # don't start backpropagating gradients until layer i_out
            if i==i_out-1:
                output.backward(g) # g should pick out the nodes in layer i_out which we want to differentiate
#### setting g to one-hot vector picks out the gradient of that one class output, right??
            ## self.input[i_layer].requires_grad = True  ###
            # note: output has shape (1, channels, spatial, spatial)
            if i<i_out-1:
                output.backward(self.input[i+1].grad.data)
                # output, for a given i, should be the output activations from that layer from forward() pass above
                # input, for a given i, should be input to a given layer, from forward() pass
            if i in self.z:
                ### print('shape of self.input[i][0] for i=' + str(i) + ":")
                ### print(self.input[i][0].data.numpy().shape)
                alpha = self.input[i].grad
                self.z[i] = (alpha > 0).float()  # ** without this line = multiplying by z, backward() reduces to propagating output target score backwards with the standard torch autograd backward function, just standard backpropagation
                self.input[i].grad = self.z[i] * alpha  # (if z remains = 1, this line does nothing
                ## print(i, self.input[i].grad.data.sum())
            if i not in self.z: ##
                print('i is not in self.z ... figure out what is going on')
                exit()
            if i==i_in:
                break
        return self.input[i_in].grad

# note that the *only* difference from VGG().make_layers() is that inplace=False instead of inplace=True
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_fb(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    """
    model = VGG_FB(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_fb_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    model = VGG_FB(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13_fb(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    """
    model = VGG_FB(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_fb_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    model = VGG_FB(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16_fb(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    """
    model = VGG_FB(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_fb_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = VGG_FB(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19_fb(**kwargs):
    """VGG 19-layer model (configuration "E")
    """
    model = VGG_FB(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_fb_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    model = VGG_FB(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
