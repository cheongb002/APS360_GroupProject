import torchvision
#from utils.settings_class import settings
import torch
import numpy as np
torch.manual_seed(0)
def getloaders(settings):
    '''
    Old version:
    transformations = torchvision.transforms.Compose([
        torchvision.transforms.Resize(settings.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    #put more transformations here with probabilities
    #Use settings parameters to determine which transformations we want
    #refer to torchvision.transforms documentation
    ])
    '''
    transformList = [
        torchvision.transforms.Resize(settings.image_size)
        ]
    
    if(settings.randomHoriFlip):
        transformList.append(torchvision.transforms.RandomHorizontalFlip(0.3)) #Parameter is probability of occurrence
    if(settings.randomVertFlip):
        transformList.append(torchvision.transforms.RandomVerticalFlip(0.3))   #Parameter is probability of occurrence
    if(settings.randomGray):
        transformList.append(torchvision.transforms.RandomGrayscale(0.3))      #Parameter is probability of occurrence
    if(settings.randomRotate and np.random.rand()>=0.3): #0.3 is probability of occurrence
        transformList.append(torchvision.transforms.RandomRotation([0,359]))   #Parameter is range of rotation
    if(settings.randomBlur and np.rand()>=0.3):
        transformList.append(torchvision.transforms.GaussianBlur(5*5, sigma=(0.1, 2.0)))

    transformList.append(torchvision.transforms.ToTensor())
    transformList.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transformations = torchvision.transforms.Compose(transformList)


    full_dataset = torchvision.datasets.ImageFolder(settings.dataset_path,transform=transformations)
    settings.classes = full_dataset.classes
    print("The following classes were found: {}".format(settings.classes))

    #split your dataset
    full_size = len(full_dataset)
    train_size = int(settings.train_val_test_split[0]*full_size)
    val_size = int(settings.train_val_test_split[1]*full_size)
    test_size = full_size - train_size - val_size
    split_sizes = [train_size,val_size,test_size]
    train_set,val_set,test_set = torch.utils.data.random_split(full_dataset,split_sizes)

    print("Number of training examples: {}".format(train_size))
    print("Number of validation examples: {}".format(len(val_set)))
    print("Number of test examples: {}".format(test_size))
    #Create dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size = 64,shuffle=True) #want largest possible bs for val set for speed
    test_loader = torch.utils.data.DataLoader(test_set,shuffle=True)

    return train_loader, val_loader, test_loader