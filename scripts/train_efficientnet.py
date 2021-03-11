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

from utils.common import *
from utils.train_utils import *
from utils.settings import settings

#import efficientnet_pytorch
from efficientnet_pytorch.model import EfficientNet

def train_efficientnet():


    efficientnet_architecture = "efficientnet_fc"

    settings = settings(use_cuda=True)
    settings.features_path = "/home/brian/Data/APS360/APS_Project/PlantVillage_Features/efficientnet-b0"
    settings.num_epochs = 20
    settings.batch_size = 256

    full_dataset = torchvision.datasets.DatasetFolder(settings.features_path,loader=torch.load,extensions=(".tensor"))
    settings.classes = full_dataset.classes
    print("Classes found: {}".format(settings.classes))

    #split full dataset
    full_size = len(full_dataset)
    train_size = int(settings.train_val_test_split[0]*full_size)
    val_size = int(settings.train_val_test_split[1]*full_size)
    test_size = full_size - train_size - val_size
    split_sizes = [train_size,val_size,test_size]

    train_features,val_features,test_features = torch.utils.data.random_split(full_dataset,split_sizes)

    print("Number of training examples: {}".format(len(train_features)))
    print("Number of validation examples: {}".format(len(val_features)))
    print("Number of test examples: {}".format(len(test_features)))

    #Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_features, batch_size=settings.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_features,batch_size = 512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_features,shuffle=True)

    model = create_model(efficientnet_architecture,settings,size=0)
    if settings.use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA available")
    else:
        print("CUDA not being used")

    #begin training
    settings.identifier = "EfficientNet_Trial1" #try to change this with each run

    print("Beginning Training")
    train_net(model,train_loader,val_loader,settings)

    test_acc = get_accuracy(model,test_loader)
    print("Final Test accuracy: {}".format(test_acc))

if __name__ == '__main__':
    train_efficientnet()