import torch
from utils.settings_class import settings
from utils.loaders import getloaders
from utils.common import create_model, get_model_name
import os

def get_classes_accuracy(model, data_loader, classes):
    n = len(classes)
    classes = [i for i in range(n)]
    correct = [0 for i in range(n)]
    total = [0 for i in range(n)]

    for imgs, labels in data_loader:
        
        #############################################
        #To Enable GPU Usage
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################
        
        output = model(imgs)
        preds = output.max(1)[1]
        for c in classes:
            correct[c] += ((preds == labels) * (labels == c)).float().sum().item()
            total[c] += max((labels == c).sum().item(), 1)

    return [i / j for i, j in zip(correct, total)]

def get_confusion_matrix(model, data_loader, classes):
    n = len(classes)
    confusion_matrix = torch.zeros(n, n, dtype=torch.int64)

    with torch.no_grad():
        for imgs, labels in data_loader:
        
            #############################################
            #To Enable GPU Usage
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################
            
            output = model(imgs)
            preds = output.max(1)[1]

            for p, t in zip(preds.view(-1), labels.view(-1)):
                confusion_matrix[p.long(), t.long()] += 1
    return confusion_matrix

def get_precision(confusion_matrix):
    return confusion_matrix.diag() / confusion_matrix.sum(1)

def get_recall(confusion_matrix):
    return confusion_matrix.diag() / confusion_matrix.sum(0)

def evaluate_model(model_type, data_loader, run_settings):
    
    model = create_model(model_type, run_settings) # change as needed (options: vgg, resnet, densenet, googlenet, resnext)
    #model.eval()
    if run_settings.use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA available")
    else:
        print("CUDA not being used")
    model_path = os.path.join(run_settings.weight_checkpoints, run_settings.identifier, get_model_name(model.name,run_settings,run_settings.num_epochs))
    state = torch.load(model_path)
    model.load_state_dict(state) #load weights in

    cm = get_confusion_matrix(model, data_loader, classes)

    plot_confusion_matrix(cm,
                          run_settings.classes,
                          title='Confusion matrix',
                          normalize=True)




if __name__ == '__main__':
    run_settings = settings()
    run_settings.identifier = "densenet_trial4"
    run_settings.use_cuda = True
    run_settings.num_epochs = 25
    run_settings.batch_size = 16
    t__, __, test_loader = getloaders(run_settings)
    evaluate_model("densenet",test_loader, run_settings)

    
