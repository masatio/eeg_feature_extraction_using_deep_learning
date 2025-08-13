import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torchvision.models import efficientnet_b2, resnet50
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def min_max_per_sample(eeg_batch):
    # eeg_signal shape: (batch_size, channels, time)
    min_vals = eeg_batch.min(dim=2, keepdim=True).values
    max_vals = eeg_batch.max(dim=2, keepdim=True).values
    return (eeg_batch - min_vals) / (max_vals - min_vals + 1e-8)

def create_heatmap(eeg_batch):

    # Apply Min-Max Normalization for each row
    eeg_batch = min_max_per_sample(eeg_batch)

    # Format heatmaps
    eeg_heatmap = torch.repeat_interleave(eeg_batch, repeats=4, dim=1)
    eeg_heatmap = eeg_heatmap.unsqueeze(1).repeat(1, 3, 1, 1)

    # Resize + Normalize heatmap using ImageNet values

    eeg_heatmap = F.interpolate(eeg_heatmap, size=(224, 224), mode='bilinear', align_corners=False)

    eeg_heatmap = (eeg_heatmap - mean)/std

    return eeg_heatmap

def create_heatmap_by_subjects(eeg_signal):
    row_min = eeg_signal.min(dim=2, keepdim=True).values
    row_max = eeg_signal.max(dim=2, keepdim=True).values

    # Apply Min-Max Normalization for each row
    eeg_signal = (eeg_signal - row_min) / (row_max - row_min + 1e-8)

    # Making it 8-bit
    #eeg_signal = (eeg_signal * 255).to(torch.int8)
    #eeg_signal = eeg_signal.float()

    # Formatting heatmaps
    heatmaps = torch.repeat_interleave(eeg_signal, repeats=4, dim=1)
    batch_size = heatmaps.size()[0]
    h = heatmaps.size()[1]
    w = heatmaps.size()[2]
    heatmaps = heatmaps.reshape((batch_size//6, 6, h, w))
    return heatmaps

class EEGToHeatmapLayer(nn.Module):
    def __init__(self):
        super(EEGToHeatmapLayer, self).__init__()

    def forward(self, eeg_signal):
        heatmap = create_heatmap(eeg_signal)
        return heatmap


class EEGEfficientNetModel(nn.Module):
    def __init__(self, n_classes=39):
        super(EEGEfficientNetModel, self).__init__()
        self.eeg_to_heatmap = EEGToHeatmapLayer()

        # Load pre-trained EfficientNet (feature extractor)
        self.feature_extractor = efficientnet_b2(weights='IMAGENET1K_V1').to(device)
        self.feature_extractor.classifier = nn.Linear(self.feature_extractor.classifier[1].in_features, n_classes)
        
        for param in self.feature_extractor.features[-2:].parameters():  # Unfreeze last 3 layers
            param.requires_grad = True


    def forward(self, eeg_signal):
        # Convert EEG signals to heatmaps
        heatmap = self.eeg_to_heatmap(eeg_signal)
        # Convert heatmaps to adequate features
        features = self.feature_extractor(heatmap)

        return features
    
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

def find_features(model, x_train, y_train, batch_size):

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []
    with torch.no_grad():
        for batch_x, _ in dataloader:  # No need for labels during feature extraction
            batch_x = batch_x.to(device)
            batch_features = model(batch_x)  # Forward pass through model
            features.append(batch_features.detach().cpu().numpy())  # Convert to numpy and detach gradients

    # Concatenate all batch features into a single array
    features = np.concatenate(features, axis=0)
    return features

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

def model_outputs(model, x, y, batch_size=128):
    x_dataset = TensorDataset(x, y)
    x_loader = DataLoader(x_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_predict = []
    with torch.no_grad():
        for inputs, labels in x_loader:
            outputs = model(inputs.to(device))
            _, predicted = outputs.max(1)
            y_predict = np.concatenate((y_predict, predicted.cpu().numpy()))

    return y_predict.reshape(-1,np.shape(y)[0])


def train_with_external_classifier(x_train, y_train, c=1000, gamma=0.0001, batch_size=128):
    # Initialize the model
    model = EEGEfficientNetModel().to(device)
    model.eval()

    # Extract features
    features = find_features(model, x_train, y_train, batch_size)
    y_train = y_train.cpu().numpy()
    # SVM
    svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=0.1))
    svm_classifier.fit(features, y_train)
    print(accuracy_score(y_train, svm_classifier.predict(features)))

    return svm_classifier, model