#general functions
from datetime import datetime
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
torch.manual_seed(1000)

from utils.models import *

def create_model(architecture, settings, size=0):
    if architecture == "efficientnet":
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained("efficientnet-b{}".format(size), num_classes=settings.num_classes())
        print("EfficientNet-b{} Model created".format(size))

        for param in model.parameters():
            param.requires_grad = False
        model._fc = nn.Sequential(
            nn.Linear(model._fc.in_features,320),
            nn.ReLU(),
            nn.Linear(320,64),
            nn.ReLU(),
            nn.Linear(64,settings.num_classes()))
        return model
    
    elif architecture == "efficientnet_fc":
        model = EfficientNet_Classifier(settings)
        print("EfficientNet Classifier created")
        return model
    
    elif architecture == "vgg":
        model = models.vgg19(pretrained=True)
        model.name = "vgg"
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096,320),
            nn.ReLU(),
            nn.Linear(320,64),
            nn.ReLU(),
            nn.Linear(64,settings.num_classes()))
        print("VGG-19 Model created")
        return model

    elif architecture == "resnet":
        model = models.resnet152(pretrained=True)
        model.name = "resnet"
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(settings.classes))
        print("ResNet-152 Model created")
        return model
    
    elif architecture == "densenet":
        model = models.densenet161(pretrained=True)
        model.name = "densenet"
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(2208,320),
            nn.ReLU(),
            nn.Linear(320,64),
            nn.ReLU(),
            nn.Linear(64,settings.num_classes()))
        print("Densenet-161 Model created")
        return model
    
    elif architecture == "googlenet":
        model = models.googlenet(pretrained=True) # requires scipy to be installed
        model.name = "googlenet"
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features,320),
            nn.ReLU(),
            nn.Linear(320,64),
            nn.ReLU(),
            nn.Linear(64,settings.num_classes()))
        print("GoogLeNet Model created")
        return model
    
    elif architecture == "resnext":
        model = models.resnext101_32x8d(pretrained=True)
        model.name = "resnext"
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features,320),
            nn.ReLU(),
            nn.Linear(320,64),
            nn.ReLU(),
            nn.Linear(64,settings.num_classes()))
        print("ResNeXt-101-32x8d Model created")
        return model
    
    else:
        print("No valid architecture was given")
        return False

def get_model_name(model_name,settings,epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
    If settings.identifier==None, then generate a name with the date, model, bs, and learning rate
    Args:
        model: model being trained on
        settings (settings object): settings of the run
        identifier (str): Optional unique identifier. Default set as the date
    Returns:
        name: A string with the hyperparameter information
    """
    if not settings.identifier:
        name = "{0}_model_{1}_bs{2}_lr{3}_epoch{4}".format(datetime.today().strftime("%b_%d_%Y"),
                                                    model_name,
                                                   settings.batch_size,
                                                   settings.learning_rate,
                                                   epoch)
    else:
        name = "{}_epoch{}".format(settings.identifier,epoch)
    return name

def get_accuracy(model, data_loader):

    correct = 0
    total = 0
    for imgs, labels in data_loader:
        
        #############################################
        #To Enable GPU Usage
        if torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################
        
        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def generate_features(data_loader,model,settings): #note batch size should be 1 here
    #loader = torch.utils.data.DataLoader(data)
    save_folder = settings.features_path
    classes = settings.classes
    use_cuda = settings.use_cuda
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    path = os.path.join(save_folder,model.name)
    if not os.path.isdir(path):
        os.mkdir(path)
    num = 0
    for imgs,labels in iter(data_loader):
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        feature = model.extract_features(imgs) #currently only configured for the EfficientNet model. Exact function may vary.
        #feature = torch.from_numpy(feature.detach().numpy())
        folder_name = os.path.join(save_folder, model.name, classes[labels])
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        torch.save(feature.squeeze(0), os.path.join(folder_name,str(num)+'.tensor'))
        num += 1
        del imgs
        del labels
    return True

