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
    run_settings.learning_rate = 1e-5 #default is 1e-3
    run_settings.identifier = "training_template"
    run_settings.settings_path = "C:\\Users\\kayef\\OneDrive\\Documents\\Year 3\\APS360 - Applied Fundamentals of Machine Learning\\trial_settings"
    run_settings.features_path = 'C:\\Users\\kayef\\OneDrive\\Documents\\Year 3\APS360 - Applied Fundamentals of Machine Learning\\PlantVillage_Features\\resnext'
    run_settings.randomRotate = True
    run_settings.randomHoriFlip = True
    run_settings.randomVertFlip = True

    train_loader, val_loader, test_loader = getloaders(run_settings)

    # create model using create_model function
    model = create_model("resnext", run_settings) # change as needed (options: vgg, resnet, densenet, googlenet, resnext)

    if run_settings.use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA available")
    else:
        print("CUDA not being used")

    print("Beginning Training")
    train_net(model, train_loader, val_loader, run_settings)

    test_acc = get_accuracy(model,test_loader)
    print("Final Test accuracy: {}".format(test_acc))

if __name__ == '__main__':
    train_pretrained()
