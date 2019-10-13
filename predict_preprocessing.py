import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session

def process_image(image_path):
    pil_image = Image.open(image_path)
    width, height = pil_image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        pil_image = pil_image.resize((round(aspect_ratio * 256), 256))
    else:
        pil_image = pil_image.resize((256, round(256 / aspect_ratio)))
    width, height = pil_image.size
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    pil_image = pil_image.crop((round(left), round(top), round(right), round(bottom)))
    np_image = np.array(pil_image) / 255
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def imshow(image, ax = None, title = None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image =  std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
                               
