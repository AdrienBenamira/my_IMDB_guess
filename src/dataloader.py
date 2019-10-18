from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob, os



class Dataloader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, phase, transform=None):
        data_dico = {}
        self.phase = phase
        if phase == "train":
            data_path = config.path.data_path_train
        if phase == "val":
            data_path = config.path.data_path_val
        if phase == "test":
            data_path = config.path.data_path_test
        os.chdir(data_path)
        for index_file, file in enumerate(glob.glob("*.png")):
            split_name = file.split(";")
            data_dico[index_file] = {'nom_file': data_path+"/"+file, "id": int(split_name[0]),
                                             "date": split_name[1],
                                             "film": split_name[2].split(("."))[0]}
        self.dico =  data_dico
        self.config = config
        self.transform = transform
        self.data_csv = pd.read_csv(config.path.data_csv, encoding='latin-1').dropna()
        self.limites = self.config.general.categorie

    def __len__(self):
        return len(self.dico)

    def __getitem__(self, idx):
        img_name = self.dico[idx]["nom_file"]
        label_idx = self.dico[idx]["id"]
        label = self.data_csv["IMDB Score"][label_idx]
        image = io.imread(img_name)
        n_label = self.categorie(label)
        #label = np.concatenate((normals,diffuse,roughness,specular),axis = 2)
        if self.transform:
            input_t = self.transform(image)
            sample = {'input': input_t, 'label': n_label}
        else:
            sample = {'input': input, 'label': n_label}
        return sample

    def categorie(self, label):
        for index_l, limit in enumerate(self.limites):
            if limit != self.limites[-1]:
                if limit<=label and label<=self.limites[index_l+1]:
                    return index_l
