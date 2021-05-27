from tqdm import tqdm
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(model, dataloader, optimizer, criterion, e):
    model.train()
    correct = 0
    total = 0
    tot_loss = 0.0
    with tqdm(dataloader, unit="batch") as tepoch:
        for item in tepoch:
            tepoch.set_description(f"Epoch {e}")
            images = item['image'].to(device)
            labels = item['label'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

        avg_loss = tot_loss/total

    return avg_loss, accuracy

def eval(model, dataloader, criterion):
    model.eval()
    correct = 0
    tot_loss = 0
    total = 0
    with torch.no_grad():
        for item in dataloader:
            images = item['image'].to(device)
            labels = item['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += len(labels)
            correct += (predicted == labels).sum().item()

        avg_loss = tot_loss / total
        accuracy = 100 * correct / total
        print('Test Acc: {}'.format(accuracy))
        print("Test Loss: {}".format(avg_loss))
        return avg_loss, accuracy




