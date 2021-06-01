import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision.models as models

from dataload import CustomImageDataset,TestImageDataset
# 0 - balancing
# 1 - inverted
# 2 - reclining
# 3 - sitting
# 4 - standing
# 5 - wheel

PATH="model-epoch-1-losses-0.26881.pth"

if __name__ == "__main__":
    # Load device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Paths
    # Data Path
    data_path = "config_data"
    # Output Path
    save_path = "outputs"
    criterion = nn.CrossEntropyLoss()
    # Data Transform

    transforms_test = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.ToTensor()])


    # Model (resnet18)
    # resnet = models.resnet18(pretrained=True)
    # resnet.fc = nn.Linear(512, 6)
    # model = resnet.to(device)

    densenet = models.densenet121(pretrained=True)
    densenet.classifier = nn.Linear(1024, 6)
    model = densenet.to(device)

    model.load_state_dict(torch.load(PATH))

    model.eval()

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
    with torch.no_grad():
        for item in test_loader:
            images = item['image'].to(device)
            index = item['index']

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # if str(predicted.item()) == "0":
            #     temp = 5
            # elif str(predicted.item()) == "1":
            #     temp = 2
            # elif str(predicted.item()) == "2":
            #     temp = 0
            # elif str(predicted.item()) == "1":
            #     temp = 1
            # elif str(predicted.item()) == "4":
            #     temp = 3
            # elif str(predicted.item()) == "5":
            #     temp = 4

            preds.append(predicted.item())
            img_ids.append(index[0].split(".")[0])

        for i in range(600):
            w.writerow([img_ids[i], preds[i]])
