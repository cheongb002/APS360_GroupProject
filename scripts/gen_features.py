'''
Generates a set of features using a model as a feature extractor
Saves the features so we don't have to regenerate them every single run
'''

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
from utils.settings_class import settings

def gen_features():

    use_cuda = True

    # Where the plant village dataset is
    dataset_path = "/home/brian/Data/APS360/APS Project/PlantVillage"

    #Where you want the features to be saved
    save_path = "/home/brian/Data/APS360/APS Project/PlantVillage_Features"

    #note the other parameter settings are not needed in this case
    # the default classes are sufficient
    run_settings = settings(dataset_path = dataset_path,
                        features_path = save_path,
                        use_cuda=use_cuda)

    '''change your settings up here'''

    transformations = torchvision.transforms.Compose([
        torchvision.transforms.Resize(run_settings.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = torchvision.datasets.ImageFolder(dataset_path,transform=transformations)

    print("The following classes were found: {}".format(classes))
    run_settings.classes = classes #just in case they're different

    data_loader = torch.utils.data.DataLoader(full_dataset)

    model = create_model("efficientnet",run_settings,size=0)

    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA is available")

    generate_features(data_loader,model,run_settings)

    print("Successfully generated features in",save_path)

if __name__ == '__main__':
    gen_features()