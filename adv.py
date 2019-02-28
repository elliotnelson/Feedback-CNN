import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import matplotlib
# % matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import itertools
from PIL import Image

import vgg
import vgg_fb
import resnet
import resnet_fb

### from scipy.io import loadmat
### x = loadmat('labels.mat')


## the feedback_model's, as currently defined, should not be fed to this method b/c x.grad gets set to None
def add_adv_noise(xx, model, epsilon, label_target=None):

    output = model(xx)
    _, label = torch.max(output, 1)
    y_true = Variable(torch.LongTensor(label), requires_grad=False)
    label = label.cpu().numpy()[0]

    loss_ftn = nn.CrossEntropyLoss()

    if label_target==None: # FGSD
        loss = -loss_ftn(output, y_true) # takes index for label, not one-hot vector
    else: # adversarial target label
        label_target = torch.tensor([label_target])
        y_target = Variable(torch.LongTensor(label_target), requires_grad=False)
        loss = loss_ftn(output, y_target)

    loss.backward(retain_graph=True) # obtain gradients on x

    x_grad = torch.sign(xx.grad.data)

    x_adv = Variable(xx.data - epsilon * x_grad)
    x_adv.requires_grad = True
    output_adv = model(x_adv)
    _, label_adv = torch.max(output_adv, 1)

    return x_adv, label_adv, label

## it's not efficient to iterate over i_out and i_in like this ... just 1 backward pass should be enough
def gradients(model, x_list, i_out_list, i_in=0, x_filter=None):

    num_x = len(x_list)
    num_io = len(i_out_list)

    shape = (num_x, num_io, 3, 224, 224)
    grads = np.zeros(shape) # will store gradients

    for i, io in itertools.product(range(num_x), range(num_io)):
    # for x, io in [(x,io) for x in x_list for io in i_out_list]:

        i_out = i_out_list[io]
        print('(i,i_out) = ' + str((i,i_out))) ####
        model.reset() # feedback_model.reset() sets the 'z' gates to 1
        output = model(x_list[i])
        # gradients of a HIDDEN layer
        g_size = model.output[i_out-1].size()
        g = torch.ones(g_size).to(device)
        model.zero_grad() ##
        gradient = feedback_model.backward(g, i_out=i_out, i_in=i_in)
        if x_filter is not None:
            gradient = torch.mul(gradient, x_filter) 
        grads[i][io] = gradient

    ### (reproduce this??:)  output[0][channel].backward(self.input[i_layer+1][0][channel]) ##
    ### output = feedback_model.output[i_out] ...

    return grads


def show_image(x):

    x = x.reshape(3,224,224).detach()
    x = np.transpose(x/2 + 0.5, (1,2,0))
    plt.imshow(x)
    plt.show()


### add PARAMETERS thru 'label_true'
image_name='images/031_324.jpeg'
#image_name='images/120_645.jpeg'
#image_filter = None
image_filter = 'images/031_crop_filter.jpeg'
label_target=None
epsilon=0.03

# Load the pretrained model
pretrained_model = models.vgg16(pretrained=True)
#pretrained_model = models.resnet18(pretrained=True)
new_model = vgg.vgg16()
#new_model = resnet.resnet18()

device = "cpu"
## device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## also commented this out in vgg_fb.py

# define useful functions
softmax = nn.Softmax(dim=0)
logsoftmax = nn.LogSoftmax(dim=0)

# Set model to evaluation mode
pretrained_model.eval()
pretrained_model = pretrained_model.to(device)
pretrained_layers = list(pretrained_model.state_dict().items())

# scaler = transforms.Scale((224, 224))
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

img = Image.open(image_name)
x = normalize(to_tensor(scaler(img))).unsqueeze(0)
x.requires_grad = True
x = x.to(device)
if image_filter is not None:
    x_filter = Image.open(image_filter)
    x_filter = to_tensor(scaler(x_filter)).unsqueeze(0)
    x_filter = x_filter.to(device)
    # see "adv_notes.py" to confirm that this filters the image correctly
else: x_filter=None

## need to understand how new_model() and feedback_model() differ from pretrained_model() 

new_model.eval()
new_model = new_model.to(device)
count=0
for key, value in new_model.state_dict().items():
    layer_name, weights = pretrained_layers[count]
    new_model.state_dict()[key].data.copy_(weights)
    count+=1
_, layer_sizes = new_model(x)
## new_model() is currently only used for layer_sizes ...
## layer_sizes computed with new_model() ... is there a simpler way to do this?
feedback_model = vgg_fb.vgg16_fb(layer_sizes=layer_sizes)
#feedback_model = resnet_fb.resnet18_fb(layer_sizes=layer_sizes)
feedback_model.eval()
feedback_model = feedback_model.to(device)
count=0
for key, value in feedback_model.state_dict().items():
    layer_name, weights = pretrained_layers[count]
    feedback_model.state_dict()[key].data.copy_(weights)
    count+=1

# get adversarial version of image and pass it through the model
x_adv, label_adv, label = add_adv_noise(x, pretrained_model, epsilon, label_target=label_target)
print('Label of clean image = ' + str(label))
print('Label of adversarial image = ' + str(label_adv))
#### model arg above ####

# forward pass x_adv through 'feedback_model' before backward pass
output_raw = feedback_model(x_adv)
#output = torch.squeeze(output_raw)
#p_output = softmax(output)
#logp_output = logsoftmax(output)

# gradients of OUTPUT layer
#g_size = output_raw.size()
#g = torch.zeros(g_size).to(device)
## label = label_adv ##
#g[0, label] = 1.0  # label is the class for which we compute gradient
#for i in range(g_size[1]):
#    g[0, i] = p_output[i] * (-1. - logp_output[i])  # this weights the output classes so we backpropagate the gradient of entropy of softmax output 

i_out_list = [i for i in range(1,len(feedback_model.output)+1)] # resnet18: should be length 10
xlist = [x, x_adv]
gradients = gradients(feedback_model, xlist, i_out_list=i_out_list, i_in=0, x_filter=x_filter)

#gradient = torch.squeeze(gradient).numpy()

# distribution of channel gradients
#gradient = np.sort(gradient)
#print(','.join([str(a) for a in gradient]))

# taking the abs value is trivial, since selective feedback pruning removed negative gradients
gradients_abs = gradients

# average the gradient over color channels 
gradients_abs_mean = np.mean(gradients_abs, 2) 

# flatten to look at shape of attention distribution
#gradient_flat = gradient_abs_mean.view(224*224)
#print(gradient_flat.size())

# average the gradient over spatial dimensions)
gradients_abs_mean = np.mean(gradients_abs_mean, 2)
gradients_abs_mean = np.mean(gradients_abs_mean, 2)
print(gradients_abs_mean)
####
print(np.divide(gradients_abs_mean[1], gradients_abs_mean[0]))

# ATTENTION MAP OF 'gradient'
image_grad = gradient.squeeze(0).permute(1, 2, 0)
gradient_map = torch.abs(image_grad).cpu().numpy()
gradient_map = np.max(gradient_map, 2)
gradient_map = gradient_map / np.max(gradient_map)
plt.subplots(1)
plt.imshow(gradient_map)
plt.show()

