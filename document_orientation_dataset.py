import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class DocumentOrientationDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        # Each image will be augmented into 4 orientations
        return len(self.images) * 4 

    def __getitem__(self, idx):
        # Determine the original image and its rotation based on index
        image_idx = idx // 4
        rotation = (idx % 4) * 90

        image_path = os.path.join(self.directory, self.images[image_idx])
        image = Image.open(image_path)

        # Rotate the image
        if rotation != 0:
            image = image.rotate(rotation)

        # Remove alpha channel by converting to RGB
        image = image.convert('RGB')
        # print(image)

        if self.transform:
            image = self.transform(image)

        # The label is determined by the rotation
        label = idx % 4

        return image, label