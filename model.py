import torch.nn as nn

class Model(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], img_size = 128, dim_output=18):
        super(Model, self).__init__()

        # TODO : build a model with three conv layers
        # number of channels goes up like : 3 -> 32 -> 64 -> 128 -> 256
        # After the conv, downsample the image with maxpool, so that image size can be reduced
        # After the maxpool, add activation (ReLU)
        # conv -> pool -> relu -> conv -> pool -> relu -> conv -> pool -> relu
        # After this, flatten the spatial feature,
        # after flattening it, add fully connected layer so that 
        # (flattened_feature) -> 128 dim feature -> 18 dim output, can be generated

        # Try to add batch norm, dropout.
        
        self.channels = channels 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.channels[0], kernel_size=3, stride=1, padding=1) #batch_num*64*64*32
        self.conv2 = nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=3, stride=1, padding=1)#32*32*64
        self.conv3 = nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=3, stride=1, padding=1)#16*16*128
        self.conv4 = nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=3, stride=1, padding=1)#8*8*256
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=dim_output)
        

    def forward(self, img):
        batch_size = img.shape[0]
        x = self.max_pool(self.activation(self.conv1(img)))
        x = self.max_pool(self.activation(self.conv2(x)))
        x = self.max_pool(self.activation(self.conv3(x)))
        x = self.max_pool(self.activation(self.conv4(x)))
        #print(x.shape)
        x = x.reshape(batch_size,-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x)) 
        x = self.activation(self.fc3(x)) 
        
        return x