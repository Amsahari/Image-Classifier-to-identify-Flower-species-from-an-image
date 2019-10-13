import argparse
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import json

from workspace_utils import active_session
from train_preprocessing import preproc
from train_model import build_model, train_model

parser = argparse.ArgumentParser()

parser.add_argument('--data_directory', action='store', default = 'flowers/', help='Set directory to load training data, e.g., "flowers"')

parser.add_argument('--save_dir', action = 'store', default = '.', dest = 'save_dir', help = 'Set directory to save checkpoints, e.g. "assets"')

parser.add_argument('--arch', action = 'store', default = 'densenet121', dest = 'arch', help = 'Set architecture, e.g. "vgg16"')

parser.add_argument('--learning_rate', action = 'store', default = 0.001, dest = 'learning_rate', help = 'Set learning rate, e.g. 0.03')

parser.add_argument('--hidden_units', action = 'store', default = 512, dest = 'hidden_units', help = 'Set no of  hidden units, e.g. 512')

parser.add_argument('--epochs', action = 'store', default = 5, dest = 'epochs', help = 'Set no of epochs, e.g. 20')

parser.add_argument('--gpu', action = 'store', default = False, dest = 'gpu', help = 'Use GPU, set switch to true')

parse_results = parser.parse_args()


data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu

image_datasets, train_loader, valid_loader, test_loader = preproc(data_dir)

model_init = build_model(arch, hidden_units)
model, optimizer, criterion = train_model(model_init, train_loader, valid_loader, learning_rate, epochs, gpu)
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'model' : model,
              'state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict,
              'criterion' : criterion,
              'epochs' : epochs,
              'class_to_idx' : model.class_to_idx}
torch.save(checkpoint, save_dir + '/checkpoint.pth')
if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + "folder"

print("Checkpoint saved to {save_dir_name}")
            
                    
                    