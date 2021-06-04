import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision.models as models

from efficientnet_pytorch import EfficientNet
from dataload import CustomImageDataset
from train import train, eval
from utils import plot

# For Colab(Gdrive mount)
# from google.colab import drive
#
# drive.mount("/content/gdrive")

if __name__ == "__main__":
    # Load device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Paths
    # Path for step1 "config_data"
    # Path for step2 "aug_data"
    # input data Path
    data_path = "aug_data"
    # Output Path
    save_path = "aug_outputs"

    # Hyper Parameters
    epoch =100
    batch_size = 8
    learning_rate = 2e-5

    # Data Transform
    transforms_train = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.ToTensor()])

    # Load Data
    train_data_set = CustomImageDataset(data_set_path=data_path+"/train", transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path=data_path+"/val", transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    if not (train_data_set.num_classes == test_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    num_classes = train_data_set.num_classes

    # Resnet152
    # resnet152 = models.resnet152(pretrained=True)
    # resnet152.fc = nn.Linear(2048, num_classes)
    # model = resnet152.to(device)
    # model.load_state_dict(torch.load(""))

    # Densenet121
    # densenet = models.densenet121(pretrained=True)
    # densenet.classifier = nn.Linear(1024, num_classes)
    # model = densenet.to(device)
    # model.load_state_dict(torch.load(""))

    # Densenet201
    # densenet = models.densenet201(pretrained=True)
    # densenet.classifier = nn.Linear(1920, num_classes)
    # model = densenet.to(device)
    # model.load_state_dict(torch.load(""))

    # Effcientnet-b0
    # model = EfficientNet.from_pretrained('efficientnet-b0')
    # model._fc = nn.Linear(in_features = 1280, out_features=num_classes)
    # model = model.to(device)
    # model.load_state_dict(torch.load(""))

    # Effcientnet-b1
    model = EfficientNet.from_pretrained('efficientnet-b1')
    model._fc = nn.Linear(in_features = 1280, out_features=num_classes)
    model = model.to(device)
    # model.load_state_dict(torch.load("b"))

    # optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training Model
    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []

    best_losses = 1e10
    name = "Adam"

    for e in range(epoch):
        # Train, Val
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, e)
        val_epoch_loss, val_epoch_acc = eval(model, test_loader, criterion)

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_acc)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_acc)

        # Plot Acc and Loss for Train and Val Results
        plot(train_accuracy, val_accuracy, train_loss, val_loss, save_path, name)

        if val_epoch_loss < best_losses:
            best_losses = val_epoch_loss
            torch.save(model.state_dict(), save_path+
                       '/epoch-{}-losses-{:.5f}.pth'.format(e + 1, val_epoch_loss))
