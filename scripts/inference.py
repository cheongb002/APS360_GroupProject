import torch
import torchvision

from utils.settings_class import settings
from utils.loaders import getloaders
from utils.common import create_model, get_model_name
from PIL import Image
import os

img_path = "/home/brian/Data/APS360/APS_Project/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"

transformList = []

transformList.append(torchvision.transforms.ToTensor())
transformList.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

transformations = torchvision.transforms.Compose(transformList)

run_settings = settings()
run_settings.identifier = "densenet_trial8"
run_settings.use_cuda = True
run_settings.num_epochs = 45



model = create_model("densenet", run_settings)
model_path = os.path.join(run_settings.weight_checkpoints, run_settings.identifier, get_model_name(model.name,run_settings,run_settings.num_epochs))

if run_settings.use_cuda and torch.cuda.is_available():
    model.cuda()
    print("CUDA available")
else:
    print("CUDA not being used")

model.eval()

with Image.open(img_path) as im:
    transformed_im = transformations(im)
    #print(type(im))
    transformed_im = torch.unsqueeze(transformed_im,0)
    transformed_im = transformed_im.cuda()
    output = model(transformed_im)
preds = output.max(1)[1]
print(int(preds))
predicted_class = run_settings.classes[int(preds)]

print("The image is of class {}".format(predicted_class))