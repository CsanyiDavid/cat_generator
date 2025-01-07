import random
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import v2

class RandomCropResizeTransform(nn.Module):
    def __init__(self, min_size, target_size, p):
        super().__init__()
        self.min_size = min_size
        self.target_size = target_size
        self.p = p #Probability of the cropping
    
    def forward(self, image):
        final_image = None
        if random.uniform(0, 1) < self.p:
            crop_size = random.randrange(self.min_size, self.target_size)
            crop_transform = v2.RandomCrop(crop_size)
            cropped_img = crop_transform(image)
            resize_transform = v2.Resize((self.target_size, self.target_size))
            final_image = resize_transform(cropped_img)
        else:
            final_image = image
        return final_image
    