import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages') # for MacBook Pro

import time
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
import os
from os import listdir
from PIL import Image

import vgg
import vgg_fb
import resnet
import resnet_fb

from fb import feedback_iters, entropy_grad_weights

### from scipy.io import loadmat
### x = loadmat('labels.mat')


def image_inputs(filenames):

    # scaler = transforms.Scale((224, 224))
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    xlist = []

    for i in range(len(filenames)):
        img = Image.open(filenames[i])
        x = normalize(to_tensor(scaler(img))).unsqueeze(0)
        x.requires_grad = True
        x = x.to(device)
        xlist.append(x)

    return xlist

def make_adv_images(model, filenames, epsilon, label_target=None, iterations=1, alpha=1):

    print('GENERATING ADVERSARIAL EXAMPLES.')

    # scaler = transforms.Scale((224, 224))
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    for i in range(len(filenames)):
        filename = filenames[i]
        if filename=='ILSVRC/Data/DET/test/ILSVRC2017_test_00000137.JPEG': ## NOT COLOR
            continue
        if filename=='ILSVRC/Data/DET/test/ILSVRC2017_test_00000657.JPEG': ## NOT COLOR
            continue
        # check if adv image already generated
        #adv_filename = 'ILSVRC_adv_vgg16_eps0pt05/' + filename[-29:-5] + '_adv.pt'
        #adv_filename = 'ILSVRC_test_samples/resnet50_FGSM_eps0pt05/' + filename[-13:-5] + '_adv.pt'
        adv_filename = 'ILSVRC_test_samples/vgg16_eps0pt05iters10alpha1_eps0pt05/' + filename[-13:-5] + '_adv.pt'
        if os.path.isfile(adv_filename):
            print('Adv image already exists')
            continue
        print('filename = ' + filename)
        img = Image.open(filename)
        x = normalize(to_tensor(scaler(img))).unsqueeze(0)
        x.requires_grad = True
        x = x.to(device)
#        if x.detach().numpy().shape[1]!=3:
#            print('Not color image. Skipping.')
#            continue
        if iterations==1:
            x_adv, label_adv, label = add_fgsm_noise(x, model, epsilon, label_target=label_target)
            print('Label of clean image = ' + str(label))
            print('Label of adversarial image = ' + str(label_adv))
        elif iterations>1:
            x_adv_list = add_iter_fgsm_noise(x, model, epsilon, iterations, alpha, label_target=label_target)
            x_adv = x_adv_list[-1]
##        adv_filename = 'ILSVRC_adv_vgg16/' + filenames[i][-29:-5] + '_adv.pt' # resnet18
##        adv_filename = 'adv_filename_temp.pt' ## filenames[i][8:14] + '_adv_vgg16.pt'
        torch.save(x_adv, adv_filename)

    return

## the feedback_model's, as currently defined, should not be fed to this method b/c x.grad gets set to None
def add_fgsm_noise(xx, model, epsilon, label_target=None, label_clean=False):

    output = model(xx)
    _, label = torch.max(output, 1)
    y_true = Variable(torch.LongTensor(label), requires_grad=False) ## CUDA?
    # label = label.cpu().numpy()[0]

    loss_ftn = nn.CrossEntropyLoss()

    if label_target is None: # FGSD
        loss = -loss_ftn(output, y_true) # takes index for label, not one-hot vector
    elif label_clean==True:
        loss = -loss_ftn(output, label_target)
    else: # adversarial target label
        label_target = torch.tensor([label_target]) ## CUDA?
        y_target = Variable(torch.LongTensor(label_target), requires_grad=False) ## CUDA?
        loss = loss_ftn(output, y_target)

    loss.backward(retain_graph=True) # obtain gradients on x

    x_grad = torch.sign(xx.grad.data)

    x_adv = Variable(xx.data - epsilon * x_grad)
    x_adv.requires_grad = True
    output_adv = model(x_adv)
    _, label_adv = torch.max(output_adv, 1)

    return x_adv, label_adv, label

def project_eps_ball(x_adv, x, epsilon):

    dx = Variable(x_adv.data - x.data) ##x_adv.sub(1.,x)
    dx = torch.clamp(dx, -epsilon, epsilon) ## better to do this as numpy operation, on x_adv.data-x.data?
 
    return Variable(dx.data + x.data) 

def add_iter_fgsm_noise(x, model, epsilon, iters, alpha=1, label_target=None):

    x_adv = x

    _, y_true = torch.max(model(x), 1) # the classification of the clean image

    print('Iterative projected attacks')

    x_adv_list = []
    for _ in range(iters):
        x_adv, label_adv, _ = add_fgsm_noise(x_adv, model, alpha, label_target=y_true, label_clean=True)
        x_adv = project_eps_ball(x_adv, x, epsilon)
        _, label_adv = torch.max(model(x_adv), 1)
        print(label_adv)
        x_adv.requires_grad = True
        x_adv_list.append(x_adv)
    
    print('Attack iterations complete')

    return x_adv_list

def crop_adv_noise(x, x_adv, x_filter, invert_filter=False):

    dx = x_adv.sub(1.,x)
    if invert_filter==True:
        x_filter = torch.add(torch.mul(x_filter, -1.), 1.)
    dx = torch.mul(dx, x_filter)
    return x.add(1.,dx)


def cosine_sim(tensor1, tensor2, spatial_filter=None):

    if spatial_filter is not None and len(tensor1.size())==4: # the 2nd condition holds for layers with spatial channels
        channels = tensor1.size()[-3]
        if tensor1.size()[-1]!=spatial_filter.size()[-1]: # downsample spatial_filter
            ratio = spatial_filter.size()[-1] // tensor1.size()[-1]
            pool = nn.AvgPool2d(ratio)
            spatial_filter = pool(spatial_filter) # downsample
        if channels!=3: # expand to number of channels
            spatial_filter = torch.mean(spatial_filter,1).unsqueeze(1) # dimension 1 should be the same as -3 (in channels def)
            spatial_filter = spatial_filter.repeat(1,channels,1,1) # expand along channel dimension
        assert tensor1.size()==spatial_filter.size()
        # project to a spatial region (e.g. bounding box)
        tensor1 = torch.mul(tensor1, spatial_filter)
        tensor2 = torch.mul(tensor2, spatial_filter)

    tensor1 = tensor1.view(-1)
    tensor2 = tensor2.view(-1)

    t11 = torch.dot(tensor1, tensor1)
    t22 = torch.dot(tensor2, tensor2)
    t12 = torch.dot(tensor1, tensor2)
    cos = t12*t12/(t11*t22)

    return np.mean(cos.data.numpy()) ## np.mean() is just supposed to convert array to number

def layers_overlap(model, x1, x2, spatial_filter=None):

    model.reset()
    _ = model(x1)
    network1 = np.copy(model.output)
    model.reset()
    _ = model(x2)
    network2 = np.copy(model.output)

    overlaps = []

    for i in range(len(network1)):
        if spatial_filter is not None and network1[i].dim()<4: # then there are no spatial components to apply spatial_filter to
            continue
        overlaps.append(cosine_sim(network1[i],network2[i], spatial_filter))

    return overlaps

def compare_clean_adv(model, filenames_clean, x_filter=None, invert_filter=False):

    # scaler = transforms.Scale((224, 224))
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    for j in range(len(filenames_clean)):

        filename = filenames_clean[j]
        num_image = filename[-13:-5]
        #adv_filename = 'ILSVRC_test_samples/vgg16_FGSM_eps0pt05/ILSVRC2017_test_' + num_image + '_adv.pt'
        adv_filename = 'ILSVRC_test_samples/vgg16_eps0pt05iters10alpha1_eps0pt05/' + num_image + '_adv.pt'
        #adv_filename = 'ILSVRC_test_samples/resnet50_FGSM_eps0pt05/' + num_image + '_adv.pt'
        # adv_filename = 'ILSVRC_adv_vgg16_eps0pt05/' + filename[-29:-5] + '_adv.pt'
        # adv_filename = 'ILSVRC_adv_resnet18_eps0pt05/' + filename[-29:-5] + '_adv.pt'
        if os.path.isfile(adv_filename)==False: # skip if adv image not found
            print('adversarial image file not found for image number ' + num_image)
            continue
        print('filename of clean image = ' + filename)
        print('filename of adv image = ' + adv_filename)

        # load clean image
        img = Image.open(filename)
        x = normalize(to_tensor(scaler(img))).unsqueeze(0)
        x.requires_grad = True
        x = x.to(device)
        # load adv image
        x_adv = torch.load(adv_filename)
        x_adv.requires_grad = True
        x_adv = x_adv.to(device)

        if x_filter is not None:
            image_filter = 'ILSVRC_test_samples/bounding_box_filters/' + num_image + '.JPEG'
            if os.path.isfile(image_filter)==False:
                print('bounding box file not found for image number' + num_image)
                continue
            print('filename for filter = ' + image_filter)
            x_filter = Image.open(image_filter)
            x_filter = to_tensor(scaler(x_filter)).unsqueeze(0)
            x_filter = x_filter.to(device)
            if invert_filter==True:
                print('Inverting x_filter.')
                x_filter = torch.add(torch.mul(x_filter, -1.), 1.)

        overlap = layers_overlap(feedback_model, x, x_adv, x_filter)
        print(overlap)

    return


## ** it's not efficient to iterate over i_out and i_in like this ... just 1 backward pass should be enough
def gradients(model, x_list, i_out_list, i_in=0, x_filter=None, g=None):

    num_x = len(x_list)
    num_io = len(i_out_list)

    if i_in==0:
        shape = (num_x, num_io, 3, 224, 224) # i_in=0
    else:
        gradshape = list(torch.squeeze(model.output[i_in-1]).shape)
        shape = tuple([num_x, num_io] + gradshape)

    grads = np.zeros(shape) # will store gradients

    for i, io in itertools.product(range(num_x), range(num_io)):
    # for x, io in [(x,io) for x in x_list for io in i_out_list]:

        i_out = i_out_list[io]
        print('i_out = ' + str(i_out))
        model.reset() # feedback_model.reset() sets the 'z' gates to 1
        output = model(x_list[i])
        # gradients of a HIDDEN layer
        g_size = model.output[i_out-1].size()
        if g is None: ## otherwise, size of g should match i_out
            gg = torch.ones(g_size).to(device) # here, we compute gradients of the *mean output* e.g. mean activation, of the layer
        else: gg = g
        model.zero_grad() ##
        gradient = feedback_model.backward(gg, i_out=i_out, i_in=i_in)
        if x_filter is not None and i_in==0:
            gradient = torch.mul(gradient, x_filter) 
        grads[i][io] = gradient.cpu() ## cuda...

    ### (reproduce this??:)  output[0][channel].backward(self.input[i_layer+1][0][channel]) ##

    return grads

def gradients_store(model, x_list, i_out_list, i_in=0, x_filter=None, g=None):

    grads = gradients(model, x_list, i_out_list, i_in, x_filter, g)

    for i in range(len(x_list)):
        g = grads[i][0] ## assume i_out_list has just 1 component
        f = open('grads_' + str(i) + '.txt', 'a')
        f.write('{')
        f.write(','.join([str(a) for a in g]))
        f.write('},\n')
        f.close()

    return

def gradient_maps(model, x, i_out_list, i_in=0, x_filter=None, g=None):

    grads = gradients(model, [x], i_out_list, i_in, x_filter, g)
    grads = grads[0] # there's just 1 image, x 

    for i in range(len(i_out_list)):
        print('gradient map for i_out = ' + str(i_out_list[i]))
        gradient = torch.from_numpy(grads[i]) ## would be better if grads was already torch Tensor
        image_grad = gradient.permute(1, 2, 0)
        gradient_map = torch.abs(image_grad).cpu().numpy()
        gradient_map = np.max(gradient_map, 2)
        gradient_map = gradient_map / np.max(gradient_map)
        plt.subplots(1)
        plt.imshow(gradient_map)
        plt.show()
 

#def ave_layer_activation():    

# compute average (positive) gradient of each layer, for clean and adversarial images
def ave_layer_grads(model, filenames_clean, filenames_adv_pt=None, i_out_list=None, grad_filter=None, invert_filter=False):

    # scaler = transforms.Scale((224, 224))
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    ave_layer_grads_clean = []
    ave_layer_grads_adv = []
    ave_layer_grads_ratio = []

    for j in range(len(filenames_clean)): # iterate over pairs of clean and adv images

        filename = filenames_clean[j]
        num_image = filename[-13:-5]
        if filenames_adv_pt is None:
            adv_filename = 'ILSVRC_test_samples/vgg16_FGSM_eps0pt05/ILSVRC2017_test_' + num_image + '_adv.pt'
            #adv_filename = 'ILSVRC_test_samples/resnet50_FGSM_eps0pt05/' + num_image + '_adv.pt'
            # adv_filename = 'ILSVRC_adv_vgg16_eps0pt05/' + filename[-29:-5] + '_adv.pt'
            # adv_filename = 'ILSVRC_adv_resnet18_eps0pt05/' + filename[-29:-5] + '_adv.pt'
        else:
            adv_filename = filenames_adv_pt[j]
        if os.path.isfile(adv_filename)==False: # skip if adv image not found
            print('adversarial image file not found for image number ' + num_image)
            continue
        print('filename of clean image = ' + filename)
        print('filename of adv image = ' + adv_filename)

        # load clean image
        img = Image.open(filename)
        x = normalize(to_tensor(scaler(img))).unsqueeze(0)
        x.requires_grad = True
        x = x.to(device)

        # load adv image
        x_adv = torch.load(adv_filename)
        x_adv.requires_grad = True
        x_adv = x_adv.to(device)

        if i_out_list==None: # initial forward pass; get list of layer numbers
            output = model(x)
            i_out_list = [i for i in range(1,len(model.output)+1)] # resnet18: should be length 10

        print('i_out_list:')
        print(i_out_list)

        # bounding box filters
        if grad_filter=='bbox':
            image_filter = 'ILSVRC_test_samples/bounding_box_filters/' + num_image + '.JPEG'
            if os.path.isfile(image_filter)==False:
                print('bounding box file not found for image number' + num_image)
                continue
        elif grad_filter is not None: # assume it's a filename
            image_filter = grad_filter
        if grad_filter is not None:
            print('filename for filter = ' + image_filter)
            x_filter = Image.open(image_filter)
            x_filter = to_tensor(scaler(x_filter)).unsqueeze(0)
            x_filter = x_filter.to(device)
            if invert_filter==True:
                x_filter = torch.add(torch.mul(x_filter, -1.), 1.)
        else:
            x_filter = None

        grads = gradients(model, [x, x_adv], i_out_list, i_in=0, x_filter=x_filter)  # recall, this is a 2D numpy array of torch arrays
        grads_mean = np.mean(grads, 2)
        grads_mean = np.mean(grads_mean, 2)
        grads_mean = np.mean(grads_mean, 2)
        grads_ratio = np.divide(grads_mean[1], grads_mean[0])
        print('grads_ratio:')
        print(grads_ratio)

        f = open("temp_ratio.txt", "a")
        ## *** (before writing many images, CUT any trivial layers)
        f.write('{')
        # f.write(str(grads_ratio[0])) # write first element, before ','
        f.write(','.join([str(r) for r in grads_ratio[1:]])) # r runs over all but first element
        f.write('},\n')
        f.close()
        #f = open("temp_adv.txt", "a")
        #f.write('{')
        ## f.write(str(grads_ratio[0])) # write first element, before ','
        #f.write(','.join([str(r) for r in grads_mean[1][1:]])) # r runs over all but first element
        #f.write('},\n')
        #f.close()

        ave_layer_grads_clean.append(grads_mean[0])
        ave_layer_grads_adv.append(grads_mean[1])
        ave_layer_grads_ratio.append(grads_ratio)

    ave_layer_grads_clean = np.asarray(ave_layer_grads_clean)
    ave_layer_grads_adv = np.asarray(ave_layer_grads_adv)
    ave_layer_grads_ratio = np.asarray(ave_layer_grads_ratio)

    # the average layer-wise attention, over all images
    ave_layer_grads_clean_mean = np.mean(ave_layer_grads_clean, axis=0)
    ave_layer_grads_adv_mean = np.mean(ave_layer_grads_adv, axis=0)
    ave_layer_grads_ratio_mean = np.mean(ave_layer_grads_ratio, axis=0)

    # variance in layer-wise attention, over all images
    ave_layer_grads_clean_std = np.std(ave_layer_grads_clean, axis=0)
    ave_layer_grads_adv_std = np.std(ave_layer_grads_adv, axis=0)
    ave_layer_grads_ratio_std = np.std(ave_layer_grads_ratio, axis=0)

    # +1 and -1 stdev
    ave_layer_grads_clean_1sig    = ave_layer_grads_clean_mean + ave_layer_grads_clean_std
    ave_layer_grads_clean_neg1sig = ave_layer_grads_clean_mean - ave_layer_grads_clean_std
    ave_layer_grads_adv_1sig      = ave_layer_grads_adv_mean   + ave_layer_grads_adv_std
    ave_layer_grads_adv_neg1sig   = ave_layer_grads_adv_mean   - ave_layer_grads_adv_std
    ave_layer_grads_ratio_1sig    = ave_layer_grads_ratio_mean + ave_layer_grads_ratio_std
    ave_layer_grads_ratio_neg1sig = ave_layer_grads_ratio_mean - ave_layer_grads_ratio_std

    print('Mean, +1sigma, and -1sigma layer-wise gradient ratios:')
    print(ave_layer_grads_ratio_mean)
    print(ave_layer_grads_ratio_1sig)
    print(ave_layer_grads_ratio_neg1sig)

    return 0 ###

def activations(model, filenames_clean, i_layer=None, filenames_adv_pt=None):

    for filename in filenames_clean:

        test_filename = 'activations_vgg16_clean/ilayer30/' + filename[-29:-5] + '.pt'
        test_filename2 = 'activations_vgg16_eps0pt05/ilayer30/' + filename[-29:-5] + '.pt'
        # skip if activations are already saved
        if os.path.isfile(test_filename) and os.path.isfile(test_filename2):
            print('Activations already saved.')
            continue 

        if filenames_adv_pt is None:
            adv_filename = 'ILSVRC_adv_vgg16_eps0pt05/' + filename[-29:-5] + '_adv.pt' # resnet18 
        else:
            print('fix this.')
            ## adv_filename = filenames_adv_pt[j]
        print('filename of clean image = ' + filename)
        print('filename of adv image = ' + adv_filename)

        # make sure image files exist
        if os.path.isfile(filename)==False:
            print('Clean image does not exist.')
            continue
        if os.path.isfile(adv_filename)==False:
            print('Adv image does not exist.')
            continue

        # load clean image
#        img = Image.open(filename)
#        x = normalize(to_tensor(scaler(img))).unsqueeze(0)
#        x.requires_grad = True
#        x = x.to(device)

        # load adv image
        x_adv = torch.load(adv_filename)
        x_adv.requires_grad = True
        x_adv = x_adv.to(device)

#        model.reset()
#        _ = model.forward(x)
#        output = model.output
#        for j in range(11,len(output)):
#            o = torch.squeeze(output[j])
#            a_filename = 'activations_vgg16_clean/ilayer' + str(j) + '/' + filename[-29:-5] + '.pt'
#            torch.save(o, a_filename)
            # o = o.detach().numpy()

        model.reset()
        _ = model.forward(x_adv)
        output_adv = model.output
        for j in range(11,len(output_adv)):
            o = torch.squeeze(output_adv[j])
            a_filename = 'activations_vgg16_eps0pt05/ilayer' + str(j) + '/' + filename[-29:-5] + '.pt'
            print(a_filename)
            torch.save(o, a_filename)
            # o = o.detach().numpy()

    return

def activations_store(model, x_list, i_layer):

    i = 0
    for x in x_list:

        model.reset()
        _ = model(x)
        o = model.output
        o = torch.squeeze(o[i_layer])
        o = o.detach().numpy()

        f = open('act_' + str(i) + '.txt', 'a')
        f.write('{')
        f.write(','.join([str(a) for a in o]))
        f.write('},\n')
        f.close()

        i += 1

    return

def show_image(x):

    x = x.reshape(3,224,224).detach()
    x = np.transpose(x/2 + 0.5, (1,2,0))
    plt.imshow(x)
    plt.show()


### add PARAMETERS thru 'label_true'
#image_names = ['ILSVRC/Data/DET/test/' + f for f in listdir('ILSVRC/Data/DET/test')]
image_names = ['ILSVRC_test_samples/images_clean/' + f for f in listdir('ILSVRC_test_samples/images_clean')]
#image_names=['images/024_727.jpeg'] # ['images/031_324.jpeg'] 
## (don't use this b/c it's not in the same order as clean images:) image_names_adv=['ILSVRC_adv_vgg16_eps0pt05/' + f for f in listdir('ILSVRC_adv_vgg16_eps0pt05')]
image_names_adv=None
#image_names_adv=['images/024_727_adv_vgg16_eps0pt5.pt'] # ['images/031_324_adv_vgg16_eps0pt5.pt']
#image_names_adv=['ILSVRC_adv_vgg16_eps0pt05/ILSVRC2017_test_00000001_adv.pt']
#image_name='images/120_645.jpeg'
#image_filter = None
#image_filter = 'center_rectangle'
#image_filter = 'images/crop_filter_center.jpeg'
#image_filter = 'ILSVRC_test_samples/bounding_box_filters/00000013.JPEG'
#image_filter = 'ILSVRC/Data/DET/test_bbox/ILSVRC2017_test_00000001_bbox.JPEG'
image_filter = 'images/031_324_crop.jpeg'
label_target=None
epsilon=0.05 # this seems to generate wrong label for most but not all images 

# Load the pretrained model
#pretrained_model = models.vgg16(pretrained=True)
pretrained_model = models.resnet50(pretrained=True)
#new_model = vgg.vgg16()
new_model = resnet.resnet50()

#device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define useful functions
softmax = nn.Softmax(dim=0)
logsoftmax = nn.LogSoftmax(dim=0)

# scaler = transforms.Scale((224, 224))
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# Set model to evaluation mode
pretrained_model.eval()
pretrained_model = pretrained_model.to(device)
pretrained_layers = list(pretrained_model.state_dict().items())

# get images as torch tensors
#xlist = image_inputs(image_names)
xlist = image_inputs(['ILSVRC_test_samples/images_clean/ILSVRC2017_test_00000013.JPEG']) 
#xlist = image_inputs(['images/031_324.jpeg'])
# get an adv image (pt file) as torch tensor
#adv_filename = 'ILSVRC_test_samples/vgg16_FGSM_eps0pt05/ILSVRC2017_test_00000017_adv.pt'
adv_filename = 'ILSVRC_test_samples/vgg16_iters10FGSM_eps0pt05/00000013_adv.pt'
#adv_filename = 'images/031_324_adv_vgg16_eps0pt5.pt' 
x_adv = torch.load(adv_filename)
x_adv.requires_grad = True
x_adv = x_adv.to(device)
xlist.append(x_adv)
x = xlist[0] ### for now

# image filter
if image_filter is 'center_rectangle': # filter that weights the center of an image
    x_filter = torch.zeros([1,3,224,224])
    for i, j, k in itertools.product(range(3), range(224), range(224)):
        x_filter[0][i][j][k] = 1. - (abs(112-j) + abs(112-k))/224
    x_filter = x_filter.to(device)
elif image_filter is not None:
    x_filter = Image.open(image_filter)
    x_filter = to_tensor(scaler(x_filter)).unsqueeze(0)
    x_filter = x_filter.to(device)
    # see "adv_notes.py" to confirm that this filters the image correctly
else:
    x_filter=None

## need to understand how new_model() and feedback_model() differ from pretrained_model() 

new_model.eval()
new_model = new_model.to(device)
count=0
for key, value in new_model.state_dict().items():
    layer_name, weights = pretrained_layers[count]
    new_model.state_dict()[key].data.copy_(weights)
    count+=1
_, layer_sizes = new_model(x)
#print(new_model.layers)
#print(layer_sizes)
## new_model() is currently only used for layer_sizes ...
## layer_sizes computed with new_model() ... is there a simpler way to do this?
#feedback_model = vgg_fb.vgg16_fb(layer_sizes=layer_sizes)
feedback_model = resnet_fb.resnet50_fb(layer_sizes=layer_sizes)
feedback_model.eval()
feedback_model = feedback_model.to(device)
count=0
for key, value in feedback_model.state_dict().items():
    layer_name, weights = pretrained_layers[count]
    feedback_model.state_dict()[key].data.copy_(weights)
    count+=1


#make_adv_images(pretrained_model, image_names, epsilon, label_target=label_target, iterations=10, alpha=1)
# x_adv_list = add_iter_fgsm_noise(x, pretrained_model, epsilon, iters=10, label_target=label_target)

# run FeedbackCNN loops
#feedback_model.reset()
#feedback_iters(x, feedback_model, iters=10, output_reinforce='doubt_label', show_attention=True)

invert_filter=False
compare_clean_adv(feedback_model, image_names, x_filter='bbox', invert_filter=invert_filter)
exit()
# get adversarial version of image and pass it through the model
#x_adv, label_adv, label = add_fgsm_noise(x, pretrained_model, epsilon, label_target=label_target)
#print('clean label:')
#print(label)
#print('adv label:')
#print(label_adv)
## x_adv, label_adv = add_iter_fgsm_noise(x, pretrained_model, epsilon, iters=10, label_target=label_target)
#### model arg above ####

# get average layer-wise gradients for many images
image_names = ['ILSVRC_test_samples/images_clean/' + f for f in listdir('ILSVRC_test_samples/images_clean')] ## ['images/031_324.jpeg']
#image_names_adv = ['images/031_324_adv_vgg16_eps0pt5.pt']
i_out_list = None ## [5,10,15,20,25,30,33,37] ####
grad_filter = 'bbox' ## 'images/031_324_crop.jpeg' ## 'bbox'
invert_filter = False
#z = ave_layer_grads(feedback_model, image_names, filenames_adv_pt=image_names_adv, i_out_list=i_out_list, grad_filter=grad_filter, invert_filter=invert_filter)
#exit()

#xlist = [x, x_adv]

#x_adv = crop_adv_noise(x, x_adv, x_filter, invert_filter=True)
#i_out_list = [21,24,27,30,33,36]
#i_out_list = [1,2,3,4,5,6,7,8,9] # resnet blocks
i_out_list=[37]
gradient_maps(feedback_model, x, i_out_list, x_filter=None)
gradient_maps(feedback_model, x_adv, i_out_list, x_filter=None)
exit()

#activations(feedback_model, image_names, filenames_adv_pt=image_names_adv)
#exit()

# forward pass x_adv through 'feedback_model' before backward pass
output = feedback_model(x)

# gradients of OUTPUT layer
g_size = output.size()
g = torch.zeros(g_size).to(device)
## label = label_adv ##
g[0, label] = 1.0  # label is the class for which we compute gradient

# get gradients / activations for a few images:
i_in=32
i_out_list = [len(feedback_model.output)] ## should make this the default, for i_out_list=None
#gradients = gradients(feedback_model, xlist, i_out_list=i_out_list, i_in=i_in, x_filter=x_filter)

gradients_store(feedback_model, xlist, i_out_list=i_out_list, i_in=i_in, x_filter=x_filter, g=g)
activations_store(feedback_model, xlist, i_in-1) # 3rd arg should be i_in-1, where i_in is arg to gradients_store
exit()

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

