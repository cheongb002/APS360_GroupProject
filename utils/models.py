import torch
import torch.nn as nn

class EfficientNet_Classifier(nn.Module):
    def __init__(self,settings):
        super(EfficientNet_Classifier, self).__init__()
        
        self.name = "EfficientNet_b0_Classifier"
        self.fc1 = nn.Linear(1280*7*7, 320) #This input dimension is specific to b0
        self.fc2 = nn.Linear(320,64)
        self.fc3 = nn.Linear(64, settings.num_classes())

    def forward(self, x):
        x = x.view(-1, 1280*7*7) #flatten feature data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x