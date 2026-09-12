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
        self.conv1 = nn.Conv2d(3, 6, 2) # params: in_channels, out_channels, kernel_size 
        self.pool = nn.MaxPool2d(2, 2) # -> params: kernel_size, stride -> we are sliding a 2*2 image across the image and keeping only the maximum
        # ... value in each window, see https://medium.com/@onally.sourour9/max-pooling-in-cnns-why-it-matters-and-how-it-works-60c590bca8b9
        self.conv2 = nn.Conv2d(6, 16, 2) # second layer 

        self.fc1 = nn.Linear(16 * 7 * 7, 120)
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

        # Additional Note for Clarity: 
        # The difference between pooling and a kernel
        # -> a convolutional filter has learnable weights which allow the 5x5 (in this case) of numbers to get better at learning weights 
        # -> pooling with a 2x2 has no learnable weights 
        # -> pooling shrinks the image since it takes the maximum of the 2x2. allows the data to become a little more simplified 

        # Why pool anyway? 
        # -> (1) It shrinks the data flowing forward, keeping the computation managable as the network keeps getting deeper
        # -> (2) It gives pixel tolerance - if an edge shifts by a pixel max pooling is likely to still catch it since we just grab the strongest value in the neighbourhood


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x)) #"how strongly was each of the 16 filters' patterns detected, at each of the 25 remaining spatial positions."
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # im assumng we dont apply the relu activation function here since its the output layer?? 
        return x 
    
net = Net()

# Define Loss function and Optimiser 

import torch.optim as optim

criterion = nn.CrossEntropyLoss() # -> network outputs logits and softmax converts these to probabilities, CEL takes the probability assigned to the correct label (class)
#                                   -> and computed loss = -ln(probability assigned to correct class)
#                                   -> Mathematical Importance: -ln(1) = 0 therefore we get a good prediction which is barely penalised 
#                                   ->  -ln(small number) ~ 1 which gets heavily penalised 
#                                   ->  Note that nn.CrossEntropyLoss() has the softmax part built in 




optimiser = optim.SGD(net.parameters(), lr = 1e-3, momentum=0.9)

# learning rate controls how much a NN adjusts it's intenal weights when training - too high and youll overfit data, too low and training takes longer 
# momentum adds a memory aspect of the previous steps - this helps to smooth noisy gradients 

# Commence with Training: 


epochs = 2
for epoch in range(epochs): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs, data is a list of: [inputs, labels]
        inputs, labels = data 

        # zero the parameter gradients 

        optimiser.zero_grad()

        # forward + backward + optimise 

        outputs = net(inputs) # -> Run the batch through: conv1→pool→conv2→pool→flatten→fc1→fc2→fc3
        loss = criterion(outputs, labels) # compute the CEL between the predicted logits and the true label for that batch 

        loss.backward() # backprop 
        optimiser.step() # paramater update 

        # Print statistics 
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 batches 
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}') 
            running_loss = 0.0 # reset the running loss at the end of each batch 

        # 50,000 training images with a batch size of four therefore one epoch is 12500 batches. we see the entire dataset twice with 2 epochs 

    print("Finished Training")

# save trained model: 

PATH = './cifar_net.pt'
torch.save(net.state_dict(), PATH)

# Test Network on the Test Data 

dataiter = iter(testloader)
images, labels = next(dataiter)

# print the images 

imshow(torchvision.utils.make_grid(images))
print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range (4)))

# load back the saved model

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(images)

# print(outputs) # for understanding clarity 
# print(classes) 

# the highest of the outputs logits matches to the class
# for example if we have [-1.0091, -1.2213,  0.7123,  1.1532,  0.3008,  0.6166, -0.0936, -0.0433, -0.1820, -0.3040], the highest value is at index 3, which matches to the 
# 3rd index of classes 

# here are the classes as a reminder: 
#     0     1       2       3       4      5       6       7        8       9 
# 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'

# so for our example the third index matches with cat, which the Ground Truth outputted 

# now make comparisons between the models prediction and the ground truth for every single batch and compute an accuracy: 

_, predicted = torch.max(outputs, 1) # get the max INDEX of the output data 
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4))) # print the predictions of the model (basically put it into a human readable form)

correct = 0
total = 0
# not training therefore no need to calculate gradients 

with torch.no_grad():
    for data in testloader:
        images, labels = data 
        outputs = net(images)

        # the prediction will be defined as the class with the highest logit out of all ten

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0) # incriment the total by the number of images in the batch, acts as a running total for the amount of images that have been processed, not batches
        correct += (predicted == labels).sum().item() # incriment the correct counter by the amount of images where it is true for predicted is the same as the label 
        

print(f'Accuracy of the network on the 10,000 images: {100 * correct // total} %')

