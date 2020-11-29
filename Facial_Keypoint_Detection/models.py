## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        #K — out_channels : the number of filters in the convolutional layer
        #F — kernel_size
        #S — the stride of the convolution
        #P — the padding
        #W — the width/height (square) of the previous layer
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # output size = (W-F)/S +1 = (224-5)/1 + 1 = 220
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers - done 
        #multiple conv layers- done
        #fully-connected layers - done
        #and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        #220/2 - (32,110,110)
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        self.conv2 = nn.Conv2d(32,64,3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        #108/2 - (64,54,54)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        self.conv3 = nn.Conv2d(64,128,3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(2, 2)
        #52/2 - (128,26,26)
        
        # fourth conv layer: 128 inputs, 256 outputs, 3x3 conv
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        self.conv4 = nn.Conv2d(128,256,3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool4 = nn.MaxPool2d(2, 2)
        #24/2 - (256,12,12)
        
        # fifth conv layer: 256 inputs, 512 outputs, 1x1 conv
        # output size = (W-F)/S +1 = (12-1)/1 + 1 = 12
        self.conv5 = nn.Conv2d(256,512,1)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool5 = nn.MaxPool2d(2, 2)
        #12/2 - (512,6,6)
        
          
        #Linear Layer
        # 512 outputs * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1024)
        
        # finally, create 136 output channels - 2 for each of the 68 keypoint (x, y) pairs
        self.fc2 = nn.Linear(1024, 136)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25)      
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))        
        x = self.pool(F.relu(self.conv2(x)))       
        x = self.pool(F.relu(self.conv3(x)))       
        x = self.pool(F.relu(self.conv4(x)))       
        x = self.pool(F.relu(self.conv5(x)))
        
        # prep for linear layer
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x