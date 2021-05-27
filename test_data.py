import csv
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataload import CustomImageDataset
from train import train, eval
from utils import plot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Data Path
data_path = "config_data"
# output path
save_path = "outputs"

# Data Transform
transforms_test = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Data Load
test_data_set = CustomImageDataset(data_set_path=data_path+"/val", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=1, shuffle=False)

num_classes = test_data_set.num_classes
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(512, num_classes)
# Load Model
model = resnet.to(device)

load = torch.load(save_path+'/model_epoch_0.pth')
model.load_state_dict(load)

f = open("./prediction.csv", "w", newline="")
w = csv.writer(f)
w.writerow(['id', 'target'])

preds = []
img_ids = []
correct = 0
with torch.no_grad():
    for (image, image_name) in test_loader:
        image = image.to(device)
        output = model(image)

        pred = output.max(1, keepdim=True)[1]
        preds.extend(pred)
        img_ids.extend(image_name)

for i in range(600):
    img_ids[i] = img_ids[i].replace('.jpg','')

for i in range(600):
    w.writerow([img_ids[i], str(preds[i].item())])

f.close()
print('save')
