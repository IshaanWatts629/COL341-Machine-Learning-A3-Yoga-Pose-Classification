### LOADING MODULES ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import LabelEncoder
import cv2
from tensorflow.keras.preprocessing import image
import torchvision.models as models
import sys

args = sys.argv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 51
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

le = LabelEncoder()

### DATALOADER ##

class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv)
        
        images = data['name'].values
        
        if self.is_train:
            labels = data['category'].values
            labels = le.fit_transform(labels)
            
        else:
            labels = None
            
        self.images = images
        self.labels = labels
    
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        img_path = self.images[idx]
        img = image.load_img(img_path, target_size=(224, 224))
        img = np.array(img)
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        img = self.img_transform(img)
        
        sample = {"images": img, "labels": label, "img_path": img_path}
        return sample

BATCH_SIZE = 32 
NUM_WORKERS = 0 

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(size=(224,112)),transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p = 0.25),transforms.ToTensor()])

# Train DataLoader

train_data = args[1]
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

### MODEL ###

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.model = models.densenet121(pretrained=True)
        
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 19)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.6)
        
    def forward(self, x):
        x = F.relu(self.model(x))
        x = F.relu(self.fc1(self.dropout1(x)))
        x = F.relu(self.fc2(self.dropout2(x)))
        x = self.fc3(x)
       
        return x

### TRAINING ###

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
num_epochs = 8

n_total_steps = len(train_loader)

train_loss = []
train_acc = []

for epoch in range(num_epochs):
    batch_loss = []
    n_correct = 0
    n_samples = 0

    for batch_idx, sample in enumerate(train_loader):
        model.train()
        
        images = sample['images']
        labels = sample['labels']
        
        images = images.to(device)
        labels = labels.to(device)
    
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        
        if (batch_idx+1)%300 == 0:
        	print(f'epoch {epoch+1}/{num_epochs}, steps {batch_idx+1}/{n_total_steps}, loss = {loss.item():.4f}')
            
    acc = 100.0 * n_correct / n_samples
    train_acc.append(acc)
    print(f'epoch {epoch+1}/{num_epochs}, accuracy = {acc}')
    
    train_loss.append(np.mean(batch_loss))

### SAVING ###

torch.save(model.state_dict(), os.path.join(args[2], 'model.pth'))

