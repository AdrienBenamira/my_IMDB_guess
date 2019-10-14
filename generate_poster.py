from utils.config import Config
import pandas as pd
import requests
import time
config = Config()


data_csv = pd.read_csv(config.path.data_csv, encoding='latin-1').dropna()
for i in range(len(data_csv.Poster)):
    try:
        url = data_csv.Poster[i]
        name_file = data_csv.Title[i][:-7].replace(" ","_")
        date = data_csv.Title[i][-5:-1]
        filename = config.path.data_posters + "/" + str(i) + ";" + str(date) + ";" + name_file +".png"
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)
        time.sleep(0.5)
    except:
        pass
