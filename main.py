from utils.config import Config
import glob, os
import torch
import matplotlib.pyplot as plt
from src import dataloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src.trainer import train_model
from src.model_conv import Net
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = Config()
writer = SummaryWriter(config.path.logs_tensorboard)
print()
print("Use Hardware : ", device)
#Reproductibilites
random.seed(config.general.seed)
np.random.seed(config.general.seed)
torch.manual_seed(config.general.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#Load data
print()
print("Load data")
trans_all = transforms.Compose([
        transforms.ToTensor()
    ])
dataload_train = dataloader.Dataloader(config, phase = "train", transform=trans_all)
dataloadered_train = DataLoader(dataload_train, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_val = dataloader.Dataloader(config, phase = "val", transform=trans_all)
dataloadered_val = DataLoader(dataload_val, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_test = dataloader.Dataloader(config, phase = "test", transform=trans_all)
dataloadered_test = DataLoader(dataload_test, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataloaders = {'train': dataloadered_train, 'val': dataloadered_val, 'test':dataloadered_test}
#Load model
print("End Load data")
print()
print("Load model")

net = Net(len(config.general.categorie) - 1)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config.train.learning_rate,
weight_decay=config.train.weight_decay)
print("End Load model")
print()
print("Start training model")
print()
best_model = train_model(config, writer, net, dataloaders, criterion, optimizer,device, num_epochs=config.train.num_epochs)
print('Finished Training')
