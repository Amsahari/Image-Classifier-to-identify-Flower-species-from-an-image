import argparse
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
import matplotlib
import matplotlib.pyplot as plt


from workspace_utils import active_session
from predict_preprocessing import process_image, imshow
from predict_model import predict

# command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--image_path', action = 'store', default = 'flowers/test/10/image_07090.jpg', help ='Path to image, ex "flowers/test/10/image_07090.jpg"')

parser.add_argument('--checkpoint', action = 'store', default ='.', dest = 'checkpoint', help = 'Directory of saved checkpoints, ex "assets"')

parser.add_argument('--topk', action = 'store', default = 5, dest = 'top_k', help = 'Return top K most likely classes, ex 5')

parser.add_argument('--category_names', action = 'store', default = 'cat_to_name.json', dest = 'category_names', help = 'file name of mapping the flower categories to real names, ex "cat_to_name.json"')

parser.add_argument('--gpu', action = 'store_true', default = False, dest = 'gpu', help = 'Use GPU, set a switch to True')
                    

parse_results = parser.parse_args()

image_path = parse_results.image_path
checkpoint = parse_results.checkpoint
top_k = int(parse_results.top_k)
category_names = parse_results.category_names
gpu = parse_results.gpu

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
filepath = checkpoint + '/checkpoint.pth'
checkpoint = torch.load(filepath)
model = checkpoint["model"]
model.load_state_dict(checkpoint['state_dict'])

np_image = process_image(image_path)
#imshow(np_image)

print("Predicting top {top_k} most likely flower names from image {image_path}.")
probs, classes = predict(np_image, model, top_k, gpu)
print(probs)
print(classes)
classes_name = [cat_to_name[i] for i in classes]
print("\n Flower name probability:")
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")
print("")


