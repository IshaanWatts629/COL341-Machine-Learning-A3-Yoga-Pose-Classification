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

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(size=(224,112)),transforms.Resize(size=(224,224)),transforms.ToTensor()])

# Test DataLoader
test_data = args[2] 
test_dataset = ImageDataset(data_csv = test_data, train=False, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

class_to_idx = {'Ardhachakrasana': 0, 'Garudasana': 1, 'Gorakshasana': 2, 'Katichakrasana': 3, 'Natarajasana': 4, 'Natavarasana': 5, 'Naukasana': 6, 'Padahastasana': 7, 'ParivrittaTrikonasana': 8, 'Pranamasana': 9, 'Santolanasana': 10, 'Still': 11, 'Tadasana': 12, 'Trikonasana': 13, 'TriyakTadasana': 14, 'Tuladandasana': 15, 'Utkatasana': 16, 'Virabhadrasana': 17, 'Vrikshasana': 18}

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

model = ConvNet().to(device)
model.load_state_dict(torch.load(os.path.join(args[1],'model.pth')))
model.eval()

### PREDICTIONS ###

idx_to_class = {}

for key in class_to_idx.keys():
    idx_to_class[class_to_idx[key]] = key

predictions = np.array([])
files = np.array([])

for batch_idx, sample in enumerate(test_loader):
    images = sample['images']
    img_path = sample['img_path']
            
    images = images.to(device)
            
    outputs = model(images)

    _, pred = torch.max(outputs, 1)
    
    predictions = np.append(predictions, pred.cpu().numpy())
    files = np.append(files, img_path)

df = pd.DataFrame(files, columns = ['name'], index = None)

category = [None for i in range(len(predictions))]

for i in range(len(predictions)):
    category[i] = idx_to_class[predictions[i]]

df['category'] = category

df.to_csv(args[3], index = None)
