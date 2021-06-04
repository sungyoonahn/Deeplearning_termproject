import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from dataload import CustomImageDataset,TestImageDataset

# Path for weight file
PATH="b1-epoch-1-losses-0.02275.pth"

if __name__ == "__main__":
    # Load device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Paths
    # Data Path
    data_path = "config_data"

    transforms_test = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.ToTensor()])


    # densenet201
    # densenet = models.densenet201(pretrained=True)
    # densenet.classifier = nn.Linear(1920, 6)
    # model = densenet.to(device)

    # Densenet121
    # densenet = models.densenet121(pretrained=True)
    # densenet.classifier = nn.Linear(1024, 6)
    # model = densenet.to(device)

    # efficientnet
    model = EfficientNet.from_pretrained('efficientnet-b1')
    model._fc = nn.Linear(in_features=1280, out_features=6)
    model = model.to(device)

    # resnet152
    # resnet152 = models.resnet152(pretrained=True)
    # resnet152.fc = nn.Linear(2048, 6)
    # model = resnet152.to(device)

    # load model weights
    model.load_state_dict(torch.load(PATH))

    model.eval()

    # open and write csv file
    f = open("./prediction.csv","w", newline = '')
    w = csv.writer(f)
    w.writerow(["id", "target"])

    transforms_test = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.ToTensor()])

    test_data_set = TestImageDataset(image_dir=data_path + "/test", transforms=transforms_test, test = True)
    test_loader = DataLoader(test_data_set, batch_size=1, shuffle=False)

    preds = []
    img_ids = []
    model.eval()
    temp = 0
    # write predictions to csv file
    with torch.no_grad():
        for item in test_loader:
            images = item['image'].to(device)
            index = item['index']

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted.item())
            img_ids.append(index[0].split(".")[0])

        for i in range(600):
            w.writerow([img_ids[i], preds[i]])
