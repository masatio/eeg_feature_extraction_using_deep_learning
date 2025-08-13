import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, n_timesteps=440, n_features=128, n_classes=39):
        super(LSTMModel, self).__init__()

        # Bidirectional LSTM layer
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=50, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.2)

        # Third LSTM layer
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=50, batch_first=True, bidirectional=False)
        self.dropout3 = nn.Dropout(0.2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_timesteps * 50, 128)  
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

# Define model, loss, and optimizer
def build_model(n_timesteps=440, n_features=128, n_classes=39, lr=0.001):
    model = LSTMModel(n_timesteps, n_features, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-5)
    return model, criterion, optimizer, scheduler

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.epochs_no_improve = 0
        self.best_model_state = None

    def __call__(self, val_acc, model):
        if self.best_score is None or val_acc > self.best_score:
            self.best_score = val_acc
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = correct / total

        print(
            f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

        scheduler.step(val_acc)

        if early_stopping(val_acc, model):
            print("Early stopping triggered. Restoring best weights.")
            break

    return model

def predict(model, criterion, optimizer, scheduler, x_val, y_val, batch_size=32):
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_outputs.append(predicted)

    all_outputs = torch.cat(all_outputs, dim=0)

    return all_outputs
