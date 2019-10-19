from utils.config import Config
import pandas as pd
import requests
import time
config = Config()
import numpy as np

np.random.seed(config.general.seed)

compteur = 0
data_csv = pd.read_csv(config.path.data_csv, encoding='latin-1').dropna()
n = len(data_csv.Poster)
liste_index = np.random.permutation(n)
for i in liste_index:
    try:
        url = data_csv.Poster[i]
        r = requests.get(url, allow_redirects=True)
        if r.status_code != 404:
            compteur += 1
            name_file = data_csv.Title[i][:-7].replace(" ","_")
            date = data_csv.Title[i][-5:-1]
            imdbid = data_csv["imdbId"][i]
            note = 10*data_csv["IMDB Score"][i]
            genre = data_csv["Genre"][i]
            name_new_file = str(compteur) + ";" + str(date) + ";" + name_file + ";" + str(imdbid) + ";" + str(note) + ";" + str(genre) +".png"
            reste = compteur % 100
            if reste <=64:
                filename = config.path.data_path_train + "/" + name_new_file
            elif 64< reste <=80:
                filename = config.path.data_path_val + "/" + name_new_file
            else:
                filename = config.path.data_path_test + "/" +name_new_file
            open(filename, 'wb').write(r.content)
        time.sleep(0.05)
    except:
        pass
    if compteur % 100 == 99:
        print(i/n)
