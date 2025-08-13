import torch 
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models import efficientnet_b2, vgg16
import model_eeg_to_visual, visual_resnet
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class GEMCIMModel(nn.Module):

    def __init__(self, eeg_input_shape, visual_input_shape, n_classes=39):
        super(GEMCIMModel, self).__init__()

        # EEG model
        self.eeg_model = model_eeg_to_visual.EEGEfficientNetModel()
        self.eeg_model.feature_extractor.classifier = nn.Linear(self.eeg_model.feature_extractor.classifier.in_features, 128)

        # Visual model
        self.visual_model = visual_resnet.ResNet50Model()
        self.visual_model.fc2 = nn.Identity()

        # Joint part of the CNN - classification layer
        self.fc = nn.Linear(256, n_classes)

    def forward(self, eeg_input, visual_input):
        x_eeg = self.eeg_model(eeg_input)
        x_visual = self.visual_model(visual_input)
        x = torch.cat((x_eeg, x_visual), dim=1)
        x = self.fc(x)
        return x
    
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


def train_model(model, criterion, optimizer, scheduler, eeg_train, visual_train, y_train, eeg_val, visual_val, y_val, batch_size=32, epochs=20):
    
    train_dataset = TensorDataset(eeg_train, visual_train, y_train)
    val_dataset = TensorDataset(eeg_val, visual_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for eeg_inputs, visual_inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(eeg_inputs.to(device), visual_inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * eeg_inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for eeg_inputs, visual_inputs, labels in val_loader:
                optimizer.zero_grad()
                outputs = model(eeg_inputs.to(device), visual_inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                val_loss += loss.item() * eeg_inputs.size(0)
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

    return model
        
def model_outputs(model, criterion, optimizer, eeg_val, visual_val, y_val, batch_size=32):

    val_dataset = TensorDataset(eeg_val, visual_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for eeg_inputs, visual_inputs, labels in val_loader:
            optimizer.zero_grad()
            outputs = model(eeg_inputs.to(device), visual_inputs.to(device))
            loss = criterion(outputs, labels.to(device))

            val_loss += loss.item() * eeg_inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    val_loss /= total
    val_acc = correct / total

    return val_acc
    