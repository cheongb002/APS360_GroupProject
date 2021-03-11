""" copied over from Brian's Lab 3b"""
#Functions specific to training

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

from utils.common import *

import progressbar


def train_net(model, train_loader, val_loader, settings):
    print(settings.num_classes())
    settings.save_settings()
    ########################################################################
    classes = settings.classes
    batch_size = settings.batch_size
    learning_rate = settings.learning_rate
    num_epochs = settings.num_epochs
    save_weights = settings.save_weights
    logdir = settings.tensorboard_logdir
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    current_loader = train_loader
    eval_loader = val_loader
    
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be SGD with Momentum.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)

    ########################################################################
    # Set up tensorboard writer to store the accuracies and losses
    
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    run_name = get_model_name(model.name,settings, settings.num_epochs)
    logdir = os.path.join(logdir, run_name)
    os.mkdir(logdir) #Will give error if directory already exists. Try giving a new unique identifier, or deleting the old run logs

    writer = SummaryWriter(log_dir=logdir)

    #iters, losses, train_acc, val_acc = [], [], [], []
    n=0
    ########################################################################
    #set up the progress bar
    widgets = [
        progressbar.Percentage(),
        progressbar.Bar(),
        ' Adaptive ', progressbar.AdaptiveETA(),
        ', ',
        progressbar.Variable('epoch'),
        ', ',
        progressbar.Variable('train_loss'),
        ', ',
        progressbar.Variable('val_accuracy', width=12, precision=12),
    ]
    

    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    #print('Set up done')
    with progressbar.ProgressBar(max_value = num_epochs,widgets = widgets) as bar:
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            #print("Epoch", epoch)
            epoch_start = time.time()
            for imgs, labels in iter(current_loader):
                #To Enable GPU Usage
                if settings.use_cuda and torch.cuda.is_available():
                
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                    ######################

                out = model(imgs)             # forward pass

                loss = criterion(out, labels) # compute the total loss
                loss.backward()               # backward pass (compute parameter updates)
                optimizer.step()              # make the updates for each parameter
                optimizer.zero_grad()         # a clean up step for PyTorch

                # save the current training information
                writer.add_scalar("loss/train", loss,n)
                

                del imgs
                del labels  # compute validation accuracy
                n+=1
            train_accuracy = get_accuracy(model,data_loader=current_loader)
            writer.add_scalar("accuracy/train",train_accuracy,epoch)

            val_accuracy = get_accuracy(model,data_loader=eval_loader)
            writer.add_scalar("accuracy/validation", val_accuracy,epoch)

            if settings.save_weights and epoch%settings.save_freq==0:
                model_path = os.join(settings.weight_checkpoints,get_model_name(model.name,settings,epoch))
                torch.save(model.state_dict(), model_path)
            
            bar.update(epoch,epoch=epoch,train_loss=loss,val_accuracy=val_accuracy)

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    torch.cuda.empty_cache()
