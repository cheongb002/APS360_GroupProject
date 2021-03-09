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
from utils import settings

use_cuda = True
dataset_path = "/home/brian/Data/APS360/APS Project/PlantVillage"

transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
full_dataset = torchvision.datasets.ImageFolder(dataset_path,transform=transformations)

classes = full_dataset.classes
print("The following classes were found: {}".format(classes))

data_loader = torch.utils.data.DataLoader(full_dataset)

model = create_model("efficientnet",classes,size=0)

if use_cuda and torch.cuda.is_available():
    model.cuda()
    print("CUDA is available")

save_path = "/home/brian/Data/APS360/APS Project/PlantVillage_Features"

#note the other parameter settings are not neededs in this case
settings = settings(classes=classes,
                    features_path = save_path,
                    use_cuda=use_cuda)


gen_features(data_loader,model,settings)

print("Successfully generated features in",save_path)