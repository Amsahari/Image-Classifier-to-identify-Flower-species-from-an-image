import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session

def predict(np_image, model, topk, gpu):
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        images = torch.from_numpy(np_image)
        images = images.unsqueeze(0)
        images = images.type(torch.FloatTensor)
        images = images.to(device)
        output = model.forward(images)
        ps = torch.exp(output)
        probs, indices = torch.topk(ps, topk)
        probs = [float(prob) for prob in probs[0]]
        inv_map = {v: k for k, v in model.class_to_idx.items()}
        classes = [inv_map[int(index)] for index in indices[0]]
    return probs, classes