#import what you need
import torch
import torchvision
from utils.settings import settings
from utils.loaders import getloaders

#set your settings, customizing from the defaults if necessary, see utils/settings.py for all parameters
settings = settings()
settings.num_epochs=100 #default is 30
settings.learning_rate = 1e-5 #default is 1e-3
settings.identifier = "training_template"
'''etc etc'''

# process dataset, with transforms if you want. 
# @Michael pls wrap these steps all the way down to the dataloader such that things like transforms 
# and training set size can simply be passed as parameters. We want to be able to call 1 function with parameters
# and get back the data_loader objects

#train_loader, val_loader, test_loader = getloaders(settings)

transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize(settings.image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #put more transformations here with probabilities
    #Use settings parameters to determine which transformations we want
    #refer to torchvision.transforms documentation
])

full_dataset = torchvision.datasets.ImageFolder(settings.dataset_path,transform=transformations)
print("The following classes were found: {}".format(classes))
settings.classes=classes

#split your dataset
full_size = len(full_dataset)
train_size = int(settings.train_val_test_split[0]*full_size)
val_size = int(settings.train_val_test_split[1]*full_size)
test_size = full_size - train_size - val_size
split_sizes = [train_size,val_size,test_size]
train_set,val_set,test_set = torch.utils.data.random_split(full_dataset,split_sizes)

print("Number of training examples: {}".format(len(train_size)))
print("Number of validation examples: {}".format(len(val_set)))
print("Number of test examples: {}".format(len(test_size)))

#Create dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings.batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set,batch_size = 512,shuffle=True) #want largest possible bs for val set for speed
test_loader = torch.utils.data.DataLoader(test_set,shuffle=True)

#create model using create_model function
model = create_model("Template model",settings)

if settings.use_cuda and torch.cuda.is_available():
    model.cuda()
    print("CUDA available")
else:
    print("CUDA not being used")



print("Beginning Training")
#train_net(model,train_loader,val_loader,settings)

test_acc = get_accuracy(model,test_loader)
print("Final Test accuracy: {}".format(test_acc))