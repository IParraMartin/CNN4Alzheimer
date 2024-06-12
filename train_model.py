import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
import torchvision.transforms as transforms

from dataset import DementiaDataset
from model import DementiaModel

DATA_PATH = 'all_images'
CSV_FILE = 'labels.csv'
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
SAVING_PATH = './'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

def get_data_loader(dataset, batch_size):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_one_epoch(model, train_dataloader, test_dataloader, criterion, optimizer, device):
    """
    We set the model to train mode. We also create these varaiables to compute performance
        - total_train_loss
        - correct_train
        - total_train
    """
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        y_hat = model(inputs)
        loss = criterion(y_hat, targets)
        total_train_loss += loss.item()

        _, predicted = torch.max(y_hat.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_acc = 100 * correct_train / total_train
    train_loss = total_train_loss / len(train_dataloader)

    """
    Now we set up the evaluation loop
    """
    model.eval()
    total_test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            y_hat_test = model(inputs)
            loss = criterion(y_hat_test, targets)
            total_test_loss += loss.item()

            _, test_predicted = torch.max(y_hat_test.data, 1)
            total_test += targets.size(0)
            correct_test += (test_predicted == targets).sum().item()
    
    test_acc = 100 * correct_test / total_test
    test_loss = total_test_loss / len(test_dataloader)

    return train_acc, train_loss, test_acc, test_loss

def train(epochs, model, train_dataloader, test_dataloader, criterion, optimizer, device):
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train_acc, train_loss, test_acc, test_loss = train_one_epoch(
            model=model, 
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device
        )
        
        print(f"Train loss: {train_loss:.2f} - Train acc: {train_acc:.2f} || Test loss: {test_loss:.2f} - Test acc: {test_acc:.2f}")
        print("-"*50)
    
    print('Training finished!')


if __name__ == '__main__':

    img_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.2786], [0.3268])
    ])

    dataset = DementiaDataset(
        csv_file='labels.csv',
        dir_to_images='all_images',
        transform=img_transform
    )

    # total dataset size = 6400
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) - train_size)
    train_split, test_split = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = get_data_loader(train_split, batch_size=BATCH_SIZE)
    test_dataloader = get_data_loader(test_split, batch_size=BATCH_SIZE)

    cnn_model = DementiaModel(
        in_channels=1, 
        n_filters=8, 
        classes=4, 
        dropout_rate=0.5
    ).to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=LEARNING_RATE)

    train(
        epochs=EPOCHS,
        model=cnn_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=loss_criterion,
        optimizer=optimizer,
        device=device
    )

    torch.save(cnn_model.state_dict(), 'dementia_detection_model.pth')