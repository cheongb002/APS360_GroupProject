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
import matplotlib.pyplot as plt


def train_net(net, train_loader, val_loader, classes, batch_size=64, learning_rate=0.01, num_epochs=30, save = False, logdir = "./logs"):
    ########################################################################
    # Train a classifier on cats vs dogs
    #target_classes = ['A','B','C','D','E','F','G','H','I']
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    current_loader = train_loader
    eval_loader = val_loader
    
    #current_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    #eval_loader = torch.utils.data.DataLoader(val_data,batch_size=1024)
    #classes = ['A','B','C','D','E','F','G','H','I']
    
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be SGD with Momentum.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    ########################################################################
    # Set up some numpy arrays to store the accuracies
    if not os.path.isdir(logdir):
        
    iters, losses, train_acc, val_acc = [], [], [], []
    n=0
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_start = time.time()
        for imgs, labels in iter(current_loader):
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              
              imgs = imgs.cuda()
              labels = labels.cuda()
              if epoch==1:
                print("successfully used CUDA on imgs")
            ######################

            out = model(imgs)             # forward pass

            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, data_loader=current_loader)) # compute training accuracy 
            val_acc.append(get_accuracy(model, data_loader=eval_loader))

            del imgs
            del labels  # compute validation accuracy
            n+=1
        #if epoch%10==0:
        #print("Epoch: {} \n Train accuracy: {} \n Val accuracy:{}".format(epoch,get_accuracy(model, data_loader=current_loader),get_accuracy(model, data_loader=eval_loader)))
        #print("Epoch {} took {}".format(epoch, time.time()-epoch_start))
    model_path = get_model_name(net.name, batch_size, learning_rate, num_epochs)
    torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    torch.cuda.empty_cache()
    print("Final Validation Accuracy: {}".format(val_acc[-1]))