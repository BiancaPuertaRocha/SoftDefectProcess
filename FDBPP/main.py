import gdown
import pandas as pd


file_id = '1t1KvW5yLhGADNlAbTygowsBfWrwogwqr'
url = f'https://drive.google.com/uc?id={file_id}'

gdown.download(url, 'local_dubbo.csv', quiet=False)

df = pd.read_csv('local_dubbo.csv')
print(df.head())
