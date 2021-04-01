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
    run_settings.identifier = "fix_testacc_1"
    name  = run_settings.identifier
    run_settings.use_cuda = True
    run_settings.save_weights = True
    run_settings.num_epochs = 1
    run_settings.batch_size = 64
    run_settings.save_freq = 5

    train_loader, val_loader, test_loader = getloaders(run_settings)
    for loader in [val_loader]: #[train_loader,val_loader,test_loader]:
        for imgs, labels in loader:
            #print(labels)
            pass
if __name__ == '__main__':
    train_pretrained()
