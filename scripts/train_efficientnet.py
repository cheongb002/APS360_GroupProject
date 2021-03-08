#goal: write a script to train a model that has features extracted from a pretrained EfficientNet B0
# subgoals: save weights
# use tensorboard for visualization
# note: Dataloader is being written by Michael, so leave that blank for now.
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

from efficientnet_pytorch import EfficientNet

get_features = True #do we want to regenerate the features

use_cuda=True
efficientnet_architecture = "efficientnet-b0"
num_classess = 
model = EfficientNet.from_pretrained(efficientnet_architecture,num_classes = num_classes)

if get_features:

    #get dataset from data loader
    #generate features, save them in a folder
