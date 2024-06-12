import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import DementiaDataset
from model import DementiaModel

DATA_PATH = 'all_images'
CSV_FILE = 'labels.csv'
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
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

        _, predicted = torch.argmax(y_hat.data, 1)
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

            _, predicted = torch.argmax(y_hat_test.data, 1)
            total_test += targets.size(0)
            correct_test += (y_hat_test == targets).sum().item()
    
    test_acc = 100 * correct_test / total_test
    test_loss = total_test_loss / len(test_dataloader)

    return train_acc, train_loss, test_acc, test_loss

def train(epochs, model, train_dataloader, test_dataloader, criterion, optimizer, device):
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train_acc, train_loss, test_acc, test_loss = train_one_epoch(model=model, 
                                                                     train_dataloader=train_dataloader, 
                                                                     test_dataloader=test_dataloader, 
                                                                     criterion=criterion, 
                                                                     optimizer=optimizer, 
                                                                     device=device)
        
        print(f"Train loss: {train_loss:.2f} - Train acc: {train_acc:.2f} || Test loss: {test_loss:.2f} - Test acc: {test_acc:.2f}")
        print("-"*50)
    
    print('Training finished!')

