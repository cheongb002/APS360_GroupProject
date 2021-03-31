#import what you need
import torch
import torchvision
from utils.settings_class import settings
from utils.loaders import getloaders
from utils.common import create_model, get_accuracy
from utils.train_utils import train_net

def train_pretrained():
    # set your settings, customizing from the defaults if necessary, see utils/settings.py for all parameters
    run_settings = settings()
    #run_settings.learning_rate = 1e-5 #default is 1e-3
    run_settings.identifier = "densenet_trial5"
    name  = run_settings.identifier
    run_settings.use_cuda = True
    run_settings.save_weights = True
    run_settings.num_epochs = 50
    run_settings.batch_size = 256
    run_settings.save_freq = 5
    run_settings.randomHoriFlip = False
    run_settings.randomVertFlip = False
    run_settings.randomRotate = True

    train_loader, val_loader, test_loader = getloaders(run_settings)
    # create model using create_model function
    model = create_model("densenet", run_settings) # change as needed (options: vgg, resnet, densenet, googlenet, resnext)

    if run_settings.use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA available")
    else:
        print("CUDA not being used")

    print("Beginning Training on", name)
    #print(name)
    train_net(model, train_loader, val_loader, run_settings,test_loader=test_loader)
    
    test_acc = get_accuracy(model,test_loader)
    print("Final Test accuracy: {}".format(test_acc))

if __name__ == '__main__':
    train_pretrained()
