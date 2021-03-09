#goal: write a script to train a model that has features extracted from a pretrained EfficientNet B0
# subgoals: save weights
# use tensorboard for visualization
# note: Dataloader is being written by Michael, so leave that blank for now.
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import efficientnet_pytorch
from efficientnet_pytorch.model import EfficientNet

from utils.common import *
from utils.settings import settings


efficientnet_architecture = "efficientnet_fc"
 
settings = settings(use_cuda=True)

model = create_model(efficientnet_architecture,settings,size=0)

full_dataset = torchvision.datasets.DatasetFolder(settings.features_path,loader=torch.load,extensions=(".tensor"))

classes = settings.classes
class_indices = {}

for letter in classes:
    class_indices[letter] = []

for i in range(len(full_dataset)):
    letter_index = full_dataset[i][1]
    letter = classes[letter_index]
    class_indices[letter].append(i)

train_indices = []
val_indices = []
test_indices = []

for letter in classes:
    n_total = len(class_indices[letter])
    n_test = int(0.05*n_total)
    n_val = int(0.15*n_total)
    test_indices += class_indices[letter][0:n_test]
    val_indices += class_indices[letter][n_test:n_val+n_test]
    train_indices += class_indices[letter][n_test+n_val:n_total]
    #print(len(train_indices))


#testing on small dataset
train_features = torch.utils.data.Subset(full_dataset, val_indices)
val_features = torch.utils.data.Subset(full_dataset, test_indices)

print("Number of training examples: {}".format(len(train_features)))
print("Number of validation examples: {}".format(len(val_features)))

#Create DataLoaders
