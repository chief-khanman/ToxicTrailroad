# -*- coding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import colorama

class CustomDataset(Dataset):
    def __init__(self, x_file, y_file, num_classes=2, transform=None):
        self.x_data = torch.load(x_file)
        self.y_data = torch.load(y_file)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]

        if self.transform:
            x = self.transform(x)
        y_onehot = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        return x, y

# Paths to Dataset files
x_file_path = '/work/flemingc/prajwal/me699/hackathon/X.pt'
y_file_path = '/work/flemingc/prajwal/me699/hackathon/y.pt'

data_transform_nth = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create an instance of the dataset
custom_dataset = CustomDataset(x_file_path, y_file_path, transform=data_transform_nth)
custom_dataset_trans = CustomDataset(x_file_path, y_file_path, transform=data_transform)
custom_dataset = torch.utils.data.ConcatDataset([custom_dataset, custom_dataset_trans])

# Split dataset into training and testing sets
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_data, test_data = random_split(custom_dataset, [train_size, test_size])


batch_size = 16
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


for i, (sample_x, sample_y) in enumerate(train_loader):
    print(f"Training Batch {i + 1} - X.shape: {sample_x.shape}, y: {sample_y}")


for i, (sample_x, sample_y) in enumerate(test_loader):
    print(f"Testing Batch {i + 1} - X.shape: {sample_x.shape}, y: {sample_y}")



custom_dataset[0][0].shape

custom_dataset[1000][1]

# Take one batch from the train loader
data, labels = next(iter(train_loader))
data, labels = data[0:5], labels[0:5]

# Plot the images
fig = plt.figure(figsize=(16, 9))
for i in range(0, 5):
    fig.add_subplot(1, 5, i + 1)
    plt.imshow(data[i].permute(1, 2, 0))

model_ft = torchvision.models.resnet18(pretrained=True)

model_ft.fc = torch.nn.Linear(in_features=512, out_features=2)
model_ft.fc

model_ft.requires_grad_(False)
model_ft.fc.requires_grad_(True)


# Commented out IPython magic to ensure Python compatibility.


def train(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs=3,
    learning_rate=0.1,
    decay_learning_rate=False,
):

    model.train()


    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.85)


    for epoch in range(num_epochs):
        print("=" * 40, "Starting epoch %d" % (epoch + 1), "=" * 40)

        if decay_learning_rate:
            scheduler.step()

        total_epoch_loss = 0.0
        # Make one pass in batches
        for batch_number, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            if batch_number % 5 == 0:
                print("Batch %d/%d" % (batch_number, len(train_loader)))

        train_acc = accuracy(model, train_loader, device)
        test_acc = accuracy(model, test_loader, device)

        print(
            colorama.Fore.GREEN
            + "\nEpoch %d/%d, Loss=%.4f, Train-Acc=%d%%, Valid-Acc=%d%%"
             % (
                epoch + 1,
                num_epochs,
                total_epoch_loss / len(train_data),
                100 * train_acc,
                100 * test_acc,
            ),
            colorama.Fore.RESET,)

def accuracy(model, data_loader, device):
    model.eval()

    num_correct = 0
    num_samples = 0
    with torch.no_grad(): 
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            predictions = torch.argmax(model(data), 1)  # find the class number with the largest output
            num_correct += (predictions == labels).sum().item()
            num_samples += len(predictions)

    return num_correct / num_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)

train(model_ft, train_loader, test_loader, device, num_epochs=4)


torch.save(model_ft.state_dict(), "./model.pth")

