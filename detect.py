import numpy as np
from random import shuffle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4096,4096)
        self.fc2 = nn.Linear(4096,1024)##,1024) ## (1000,512)
        self.fc3 = nn.Linear(1024,64) ## (4096,256)
        self.fc4 = nn.Linear(64,12)
        self.fc5 = nn.Linear(12,2)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = self.dropout(x)
        x = self.fc5(x)
        return x

## ok, dropout always seems to cause the loss to get stuck at ~0.69something

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # currently these parameters assume input shape [1,512,14,14]
        self.conv1 = nn.Conv2d(512, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        # currently these parameters assume input shape [1,512,14,14]
        self.conv1 = nn.Conv2d(512, 512, 3)
        self.conv2 = nn.Conv2d(512, 128, 2)
        self.apool = nn.AvgPool2d(4,4) # compresses 14x14 to 3x3
        self.pool = nn.MaxPool2d(6, 6)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 2)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.apool(x) ## including this first doesn't seem to speed it up; why does CNN2() take much longer than Net()? only the apool() step looks like it could possibly take longer...
        x = F.relu(self.conv2(x))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        #x = F.relu(self.conv1(x))
        #x = self.pool(F.relu(self.conv1(x)))
        #x = x.view(-1, 512 * 2 * 2)
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return x

# load data for clean and adversarial classes
def load_data(dir_clean, dir_adv, subrange):

    for _,_,f_names in os.walk(dir_clean):
        filenames_clean = f_names
    # the following should be the same as filenames_clean
    #for _,_,f_names in os.walk(dir_adv):
    #    filenames_adv = f_names

    inputs = []

    print('Loading data.')

    for f in filenames_clean[subrange[0]:subrange[1]]:
        print(f)
        a = torch.load(dir_clean + f)
        if os.path.isfile(dir_adv + f)==False:
            continue
        a_adv = torch.load(dir_adv + f) # assumes same filenames in dir_clean and dir_adv
        a = a.unsqueeze(0)
        a_adv = a_adv.unsqueeze(0)
        # print(a.shape)
        inputs.append((a, a_adv))

    shuffle(inputs)

    inputs = [x for pair in inputs for x in pair] # checked that this set inputs=[clean1,adv1,clean2,adv2,...]
 
    print('Num loaded images')
    print(len(inputs))

    label = torch.zeros((1,1), dtype=torch.long)
    label_adv = torch.ones((1,1), dtype=torch.long)
    labels = []
    for _ in range(len(inputs)//2):
        labels += [label, label_adv]

    # shuffle
    data = list(zip(inputs, labels))
    shuffle(data)
    inputs, labels = zip(*data)

    #inputs = torch.cat(inputs, dim=0)
    #labels = torch.cat(labels)

    #print('load_data: inputs and labels shapes:')
    #print(inputs.shape)
    #print(labels.shape)

    return inputs, labels


def test(net, inputs0, labels0, minibatch_size=200):

    inputs0 = torch.cat(inputs0, dim=0)
    labels0 = torch.cat(labels0)

    criterion = nn.CrossEntropyLoss()

    D = len(inputs0)
    m = minibatch_size

    loss_list = []

    for i in range(D//m):

        # get the data
        inputs = inputs0[m*i:m*(i+1)]
        labels = labels0[m*i:m*(i+1)]

        inputs = inputs.to(device)
        labels = labels.to(device)

        labels = labels.squeeze() ## ACTUALLY, I can dispense of this if I replace (1,1) above in label def with 1, I think

        outputs = net(inputs)

        print('outputs:')
        print(outputs) ####

        loss = criterion(outputs, labels)

        loss_list.append(loss.item())
        print('minibatch TEST loss list:')
        print(loss_list)

    return sum(loss_list)/len(loss_list)


def train(net, inputs0, labels0, minibatch_size=300, inputs_test=None, labels_test=None):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=0.01)
    # optimizer = optim.RMSprop(net.parameters(), lr=0.01, weight_decay=0.02, momentum=0.7)

    #for w in net.parameters():
    #    torch.nn.init.normal(w,mean=0,std=0.0001)

    D = len(inputs0)
    m = minibatch_size

    print('Training set size:')
    print(D)

    epoch_loss_list, loss_test_list = [], []

    for epoch in range(1000):  # loop over the dataset multiple times

        loss_list = []

        print('Starting epoch #:')
        print(epoch+1)

        # shuffle
        data = list(zip(inputs0, labels0))
        shuffle(data)
        inputs0, labels0 = zip(*data)

        inputs1 = torch.cat(inputs0, dim=0)
        labels1 = torch.cat(labels0)

        running_loss = 0.0
        for i in range(D//m+1): ## +1?

            if m*i>=D: # end of training set
                break
            print(i)

            # get the data
            inputs = inputs1[m*i:min(m*(i+1),D)]
            labels = labels1[m*i:min(m*(i+1),D)]

            inputs = inputs.to(device)
            labels = labels.to(device)

            labels = labels.squeeze() ## ACTUALLY, I can dispense of this if I replace (1,1) above in label def with 1, I think

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            #print(inputs.shape)
            #print(outputs.shape)
            #print(labels.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(loss.is_cuda) # confirm using GPU

            # print statistics
            ##running_loss += loss.item()
            ##if i % 20 == 19:    # print every 200 mini-batches
            ##    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
            ##    running_loss = 0.0
            loss_list.append(loss.item())
            #print('minibatch loss list:')
            #print(loss_list)
        epoch_loss_list.append(sum(loss_list)/len(loss_list))
        print('epoch loss list:')
        print(epoch_loss_list)

        # print weights
        #for param in net.parameters():
        #    print(param.data)

        loss_test = test(net, inputs_test, labels_test)
        loss_test_list.append(loss_test)
        print('test loss list:')
        print(loss_test_list)

####        torch.save(net.state_dict(), 'saved_networks/ilayer29/epoch' + str(epoch) + '.pt')
        # load model
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#net = Net()
net = CNN2()
net = net.to(device)

dir_clean = 'activations_vgg16_clean/ilayer29/'
dir_adv = 'activations_vgg16_eps0pt05/ilayer29/'

# range of files for train and test images
train_range = (0,1600) ####0)
test_range = (1601,2000)

inputs, labels = load_data(dir_clean, dir_adv, train_range)
inputs_test, labels_test = load_data(dir_clean, dir_adv, test_range)

#print(inputs.get_device())

train(net, inputs, labels, inputs_test=inputs_test, labels_test=labels_test)

