import torch

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