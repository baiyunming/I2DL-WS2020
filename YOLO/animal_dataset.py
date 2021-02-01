import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision

class AnimalDataset(Dataset):
    def __init__(self, csv_file, S=5, B=2, C=4):
        self.S = S
        self.B = B
        self.C = C
        self.csv_file = pd.read_csv(csv_file)

        # Add some tranforms for data augmentation.
        self.tensor_transform = torchvision.transforms.ToTensor()
        self.normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
        self.resize = torchvision.transforms.Resize(size=(224, 224))
        self.transform = torchvision.transforms.Compose([self.resize,
                                                         self.tensor_transform,
                                                         self.normalize_transform])

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img = Image.open(self.csv_file.iloc[idx, 0])
        img_tensor = self.transform(img)

        # txt file --> label boxes
        boxes = []
        with open(self.csv_file.iloc[idx, 1]) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in
                                                    label.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)

            # S cells in x and y direction
            i = int(self.S * y)
            j = int(self.S * x)
            # center of box relative to the boundary of cell (i,j)
            x_cell = self.S * x - j
            y_cell = self.S * y - i
            width_wrt_cell = self.S * width
            height_wrt_cell = self.S * height

            label_matrix[i, j, class_label] = 1
            # exist object = 1
            label_matrix[i, j, 4] = 1
            # x, y, width, height
            label_matrix[i, j, 5] = x_cell
            label_matrix[i, j, 6] = y_cell
            label_matrix[i, j, 7] = width_wrt_cell
            label_matrix[i, j, 8] = height_wrt_cell

        return img_tensor, label_matrix

# def test():
#     batch_size = 8
#     train_dataset = AnimalDataset("F:\Jupyter-Notebook\I2DL_WS20\ObjectDetection\\animal_dataset\\train.csv")
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     img, label = next(iter(train_dataloader))
#     print(img.shape)
#     print(label.shape)
#
# test()