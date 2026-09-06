# Development of a convolutional neural network using PyTorch after learning some of the basics of PyTorch

# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html # 

# steps: 
# -> Load and normalise the training and test datasets using torchvision
# -> Define a convolutional neural network 
# -> Define a loss function 
# -> Train the network on the training data 
# -> Test the network on the test data 


## Imports ## 
import torch
import torchvision
from torchvision.transforms import v2 
## ~~~~~~~ ##                     

transform = v2.Compose([ # compose allows several transforms to be composed together as one 
    v2.ToImage(), # makes pytorch treat the numbers as a picture and not just a load of random values 
    v2.ToDtype(torch.float32, scale=True), # changing the data type 
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalisation, first parameter is the mean and the second is the std 

    # note that we are computing (x - 0.5) / 0.5 per pixel 


    # basically we are just shifting the [0,1] range to [-1, 1]
    # why are we taking three inputs and not two? CIFAR10 is RGB, so normalise takes a tuple of 3 means and 3 stds 
])

batch_size = 4 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0) # numworkers means things are not sequential and they happen at the same time 


# condensing into one class the whole process of fetching the files, unpacking them and then giving an indexable object
# if we call trainset[i] we will return (image_tensor, label), note this has already been ran through the transform sequence we defined

# Dataset: 'Heres how to get one item'
# DataLoader: 'Heres how to serve up batches of items during training'
# Dataloader replaces the manual batching I would otherwise need to do myself 

### What is the point of shuffling at all? ###
# -> Each epoch needs to see the data in a different random order, if it didn't, the model would see the same batches every epoch 
# -> and would pick up on these patterns 

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# visualisation of some of the training images: 

import matplotlib.pyplot as plt 
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # undo the normalisation so that imshow doesnt throw an error. if we have values of -0.8 for example, we will get an error
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # input here is (H, W, C)
    plt.show()

# access random training images 

    
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images)) # images is a whole batch with shape (batch_size, 3, 32, 32)
print(' '.join(f'{classes[labels[j]]:5s}' for j in range (batch_size))) # loop over every image in the batch 
# join is a concatenation operator 

# Define the Network # 

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # params: in_channels, out_channels, kernel_size 
        self.pool = nn.MaxPool2d(2, 2) # -> params: kernel_size, stride -> we are sliding a 2*2 image across the image and keeping only the maximum
        # ... value in each window, see https://medium.com/@onally.sourour9/max-pooling-in-cnns-why-it-matters-and-how-it-works-60c590bca8b9
        self.conv2 = nn.Conv2d(6, 16, 5) # second layer 

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # BREAKDOWN # 
        # in_channel = 3 because we have 3 different channels, RGB
        # out_channel is purely a hyperparameter that works well
        # kernel_size = 5 gives us a 5x5 window or filter which we place over the 32*32. this means one filter is covering
        # about 2.4% of the images area at a time ## EXPERIMENT WITH THIS AT THE END ## 

        # Each output channel is one learned filter and each filter learns todetect one type of pattern - not hand chosen 
        # we use a small amount of learned filters at this layer and then more on the next layer. This is because at the earliest stage
        # images only really have a few genuinely distinct low level patterns worth detecting such as horizontal and vertical edges

        # we use more later on to pick up on higher level features 

        # maxpool basic idea: When you look at a picture, your brain doesn’t focus on every single pixel,
        # it zooms in on what matters: shapes, patterns, and key features.




    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x)) #"how strongly was each of the 16 filters' patterns detected, at each of the 25 remaining spatial positions."
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # im assumng we dont apply the relu activation function here since its the output layer?? 

net = Net()
