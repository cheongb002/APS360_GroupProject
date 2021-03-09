#general functions
import os
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
torch.manual_seed(1000)

def create_model(architecture, classes, size=0):
    if architecture == "efficientnet":
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained("efficientnet-b{}".format(size), num_classes=len(classes))
        print("EfficientNet-b{} Model created".format(size))
        return model
    else:
        print("No valid architecture was given")
        return False

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

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

def gen_features(data_loader,model,settings): #note batch size should be 1 here
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
        feature = model.extract_features(imgs)
        #feature = torch.from_numpy(feature.detach().numpy())
        folder_name = os.path.join(save_folder, model.name, classes[labels])
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        torch.save(feature.squeeze(0), os.path.join(folder_name,str(num)+'.tensor'))
        num += 1
        del imgs
        del labels
    return True

