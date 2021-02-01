import torch.nn as nn
import torch
import torchvision

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolo1(nn.Module):
    def __init__(self, in_channels = 3, S =5, B =2, C = 4, **kwargs):
        super(Yolo1, self).__init__()

        self.in_channels = in_channels
        self.darknet = nn.Sequential(
            CNNBlock(in_channels= self.in_channels, out_channels=64, kernel_size = 7, stride = 2, padding = 3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            CNNBlock(in_channels=64, out_channels=192, kernel_size=3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            CNNBlock(in_channels=192, out_channels=128, kernel_size=1, stride = 1, padding = 0),
            CNNBlock(in_channels=128, out_channels=256, kernel_size = 3, stride = 1, padding = 1),
            CNNBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        )

        #according the original paper
        self.S = S
        self.B = B
        self.C = C

        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.S * self.S * (self.C + self.B * 5)),
        )

    def forward(self, x):
        x = self.darknet(x)
        print("output dimension after darknet")
        print(x.shape)
        return self.fcs(torch.flatten(x, start_dim=1))


class YOLO_Resnet(nn.Module):
    def __init__(self, feat_dim=2048, S=5, C=4, B=2):
        super(YOLO_Resnet, self).__init__()

        self.feat_dim = feat_dim
        self.S = S
        self.C = C
        self.B = B

        self.backbone = torchvision.models.resnet50(pretrained=True)

        # # Fix Initial Layers
        for p in list(self.backbone.children())[:-2]:
            p.requires_grad = False

        # # get the structure until the Fully Connected Layer
        modules = list(self.backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # Add new fully connected layers
        self.fc1 = nn.Linear(feat_dim, 1024)  # 2048 -> 1024
        self.fc2 = nn.Linear(1024, self.S * self.S * (self.C + self.B * 5))  # 1024 -> 5*5*(4+2*5)
        self.dropout = nn.Dropout(p=0.5)

        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.1)

        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, img):
        batch_size = img.shape[0]
        out = self.backbone(img)  # get the feature from the pre-trained resnet
        out = self.dropout(self.activation(self.fc1(out.view(batch_size, -1))))
        out = self.fc2(out)

        return out




# def test():
#     device = 'cuda'
#     model = YOLO_Resnet().to(device)
#     x = torch.randn((2,3,224,224))
#     x = x.to(device)
#     print(model(x).shape)
#
#
# test()