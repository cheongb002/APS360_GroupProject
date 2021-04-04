#import what you need
import torch
import torchvision
from utils.settings_class import settings
from utils.loaders import getloaders
from utils.common import create_model, get_accuracy,get_model_name
from utils.train_utils import train_net
import os 
def train_pretrained():
    # set your settings, customizing from the defaults if necessary, see utils/settings.py for all parameters
    run_settings = settings()
    #run_settings.learning_rate = 1e-5 #default is 1e-3
    run_settings.identifier = "densenet_trial8"
    name  = run_settings.identifier
    run_settings.use_cuda = True
    run_settings.save_weights = True
    run_settings.num_epochs = 45
    run_settings.batch_size = 256

    train_loader, val_loader, test_loader = getloaders(run_settings)
    model = create_model("densenet", run_settings) # change as needed (options: vgg, resnet, densenet, googlenet, resnext)
    #model.eval()
    if run_settings.use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA available")
    else:
        print("CUDA not being used")
    model_path = os.path.join(run_settings.weight_checkpoints, run_settings.identifier, get_model_name(model.name,run_settings,run_settings.num_epochs))
    state = torch.load(model_path)
    model.load_state_dict(state)
    acc = get_accuracy(model,test_loader)
    print(acc)
if __name__ == '__main__':
    train_pretrained()