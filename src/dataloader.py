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
            print(split_name)
            if len(split_name) == 6:
                compteur = int(split_name[0])
                date = split_name[1]
                name_film = split_name[2]
                imdbid = int(split_name[3])
                note = float(split_name[4])/10
                genre = split_name[5]
                data_dico[index_file] = {
                'nom_file': data_path+"/"+file,
                "compteur": int(compteur),
                "date": date,
                "imdbid": imdbid,
                "note" : note,
                "genre" : genre,
                "name_film": split_name[2],
                }
        self.dico =  data_dico
        self.config = config
        self.transform = transform
        self.data_csv = pd.read_csv(config.path.data_csv, encoding='latin-1').dropna()
        self.limites = self.config.general.categorie

    def __len__(self):
        return len(self.dico)

    def __getitem__(self, idx):
        img_name = self.dico[idx]["nom_file"]
        image = io.imread(img_name)
        label = self.dico[idx]["note"]
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
