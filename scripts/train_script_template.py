#import what you need
import torch
import torchvision
import os
ROOT_DIR = os.path.abspath(os.curdir)
#utils_module = os.path.join(ROOT_DIR,'utils')
import sys
if ROOT_DIR not in sys.path:
    print("path appended to system path")
    sys.path.append(ROOT_DIR)
import utils.settings_class
from utils.settings_class import settings

assert False
def train_template():

    #set your settings, customizing from the defaults if necessary, see utils/settings.py for all parameters
    run_settings = settings()
    run_settings.num_epochs=100 #default is 30
    run_settings.learning_rate = 1e-5 #default is 1e-3
    run_settings.identifier = "training_template"
    '''etc etc'''

    # process dataset, with transforms if you want. 
    # @Michael pls wrap these steps all the way down to the dataloader such that things like transforms 
    # and training set size can simply be passed as parameters. We want to be able to call 1 function with parameters
    # and get back the data_loader objects

    # train_laoder, val_l, test_l = generate_loaders(settings)

    transformations = torchvision.transforms.Compose([
        torchvision.transforms.Resize(run_settings.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = torchvision.datasets.ImageFolder(run_settings.dataset_path,transform=transformations)
    print("The following classes were found: {}".format(classes))
    run_settings.classes=classes

    #split your dataset
    full_size = len(full_dataset)
    train_size = int(run_settings.train_val_test_split[0]*full_size)
    val_size = int(run_settings.train_val_test_split[1]*full_size)
    test_size = full_size - train_size - val_size
    split_sizes = [train_size,val_size,test_size]
    train_set,val_set,test_set = torch.utils.data.random_split(full_dataset,split_sizes)

    print("Number of training examples: {}".format(len(train_size)))
    print("Number of validation examples: {}".format(len(val_set)))
    print("Number of test examples: {}".format(len(test_size)))

    #Create dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run_settings.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size = 512,shuffle=True) #want largest possible bs for val set for speed
    test_loader = torch.utils.data.DataLoader(test_set,shuffle=True)


    #create model using create_model function
    model = create_model("Template model",run_settings)

    if run_settings.use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA available")
    else:
        print("CUDA not being used")



    print("Beginning Training")
    #train_net(model,train_loader,val_loader,settings)

    test_acc = get_accuracy(model,test_loader)
    print("Final Test accuracy: {}".format(test_acc))
if __name__ == '__main__':
    train_template()
