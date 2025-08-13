import random

import numpy as np
import torch
import torch.nn as nn
from pyexpat import features
from sklearn.model_selection import train_test_split, StratifiedKFold
import model_eeg_to_visual
import eeg_lstm_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from model_eeg_to_visual import EEGEfficientNetModel
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load(r"C:\Users\masat\Downloads\eeg_14_70_std.pth")
dataset = data['dataset']
labels = data["labels"]
images = data["images"]

###############
# Dataset se sastoji iz EEG, rednog broja slike, odgovarajuce oznake i rednog broja subjekta
# Labels su sve postojece labele, pretpostavljam da su tim redom numerisane
# Images bi trebalo da bude 39, klasu 33 treba da uklonim

#### Pretprocesiranje podataka ####
# 1. Moram da uklonim klasu 33
# 2. Uzeti 440 odbiraka za svaki od primera
# 3. Z-skor ---- ovo nakon podele na trening, validacioni i test skup!
# 4. Zabeleziti u posebni niz labele

dataset_eeg = [d for d in dataset if d['label'] != 33] # 1. korak radi
true_labels = []
true_images = []

dataset_eeg = sorted(dataset_eeg, key=lambda d: d['image'])

for d in dataset_eeg:
    if d['label'] == 39:
        d['label'] = 33
    d['eeg'] = d['eeg'][:,20:460] # 2. korak radi
    true_labels.append(d['label'])
    true_images.append(d['image'])

all_eeg = np.concatenate([sample['eeg'][np.newaxis, ...] for sample in dataset_eeg], axis=0)  # shape: (N, C, T)
channel_means = all_eeg.mean(axis=(0, 2), keepdims=True)  # shape: (1, C, 1)
channel_stds = all_eeg.std(axis=(0, 2), keepdims=True)    # shape: (1, C, 1)

for sample in dataset_eeg:
    sample['eeg'] = (sample['eeg'] - channel_means[0]) / channel_stds[0]

unique_pairs = list(set(zip(true_images, true_labels)))
true_images = [pair[0] for pair in unique_pairs]
true_labels = [pair[1] for pair in unique_pairs]

images_train, images_test, labels_train, labels_test = train_test_split(true_images, true_labels, test_size=0.15, random_state=42, stratify=true_labels)
images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=0.2142857, random_state=42, stratify=labels_train)

#### Podela podataka na obucavajuci, test i validacioni skup ####
# Odnos u radu je 70:15:15, 5. korak radi!

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

for d in dataset_eeg:
    if d['image'] in images_train:
        X_train.append(d['eeg'])
        y_train.append(d['label'])
    elif d['image'] in images_val:
        X_val.append(d['eeg'])
        y_val.append(d['label'])
    else:
        X_test.append(d['eeg'])
        y_test.append(d['label'])

X_train = torch.stack(X_train, dim=0)#.to(device)
y_train = torch.tensor(y_train)#.to(device)
X_test = torch.stack(X_test, dim=0)#.to(device)
y_test = torch.tensor(y_test)#.to(device)
X_val = torch.stack(X_val, dim=0)#.to(device)
y_val = torch.tensor(y_val)#.to(device)


model = EEGEfficientNetModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, min_lr=1e-5)
#model, train_acc, val_acc = model_eeg_to_visual.train_model(model, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, 64, 200)
#torch.save(model, 'C:/users/masat/Desktop/Projekat iz Neuralnih/efficientnetb2_finetuned.pth')
model = torch.load('C:/users/masat/Desktop/Projekat iz Neuralnih/efficientnetb2_finetuned.pth', weights_only=False)

y_test_pred = model_eeg_to_visual.model_outputs(model, X_test, y_test, 32)
y_test_pred = np.array(y_test_pred).reshape(-1)

print(accuracy_score(y_test.cpu().numpy(), y_test_pred))
cm = confusion_matrix(y_test.cpu().numpy(), y_test_pred)
plt.figure()
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.show()
