import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from torchvision.models import resnet50
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


class ResNet50Model(nn.Module):
    def __init__(self, n_classes=39):
        super(ResNet50Model, self).__init__()

        # Load pre-trained ResNet50
        self.resnet = resnet50(pretrained=True).to(device)
        self.resnet.fc = nn.Identity()
        
        for param in self.resnet.layer3.parameters():  # Unfreeze last 3 layers
            param.requires_grad = True
            
        # Global Average Pooling
        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # Dropout for preventing overfitting
        self.dropout1 = nn.Dropout(0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 128)  # ResNet-50 outputs 2048 channels
        self.relu = nn.ReLU()
        #self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, n_classes)  # Final classification layer
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)
        #x = self.global_avg_pool(x)
        x = torch.flatten(x,1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

def build_model(n_classes=39, lr=0.001):
    model = ResNet50Model(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, min_lr=1e-5)
    return model, criterion, optimizer, scheduler

class EarlyStopping:
    def __init__(self, patience=10, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.epochs_no_improve = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_score is None or val_loss < (self.best_score - 1e-4):

            self.best_score = val_loss
            self.epochs_no_improve = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                return True  # Stop training
        return False


def train_model(model, criterion, optimizer, scheduler, x_train, y_train, x_val, y_val, batch_size=32, epochs=20):
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.to(device)).sum().item()

        val_loss /= total
        val_acc = correct / total

        print(
            f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

        scheduler.step(val_loss)

        if early_stopping(val_loss, model):
            print("Early stopping triggered. Restoring best weights.")
            break

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()
    train_acc = correct / total

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()
    val_acc = correct / total
    
    return model, train_acc, val_acc