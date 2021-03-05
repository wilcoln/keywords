import os
import urllib.request
import zipfile
from collections import defaultdict
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'download')

def table():
    return pd.read_csv(os.path.join(BASE_DIR, 'datasets.csv'))

def download(dataset):
    
    # Create download dir if not exists
    try:
        os.stat(DOWNLOAD_DIR)
    except:
        os.mkdir(DOWNLOAD_DIR) 
    
    # Download zip file
    print(f'Downloading {dataset}.zip...')
    dataset_zip_path = f'{os.path.join(DOWNLOAD_DIR, dataset)}.zip'
    urllib.request.urlretrieve(_url_of(dataset), dataset_zip_path)
    
    # Extract zip file
    print(f'Extracting {dataset}.zip...')
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.filename = dataset
        zip_ref.extractall(DOWNLOAD_DIR)
        
    # Delet zip file
    print(f'Deleting {dataset}.zip...')
    os.remove(dataset_zip_path)
    

def load(dataset=None, corpora=None):
    data = defaultdict()

    dataset_path = os.path.join(DOWNLOAD_DIR, dataset)
    
    # Download if necessary
    if not os.path.exists(dataset_path):
        download(dataset)

    # Load language
    data['lang'] = {}
    try:
        with open(f'{dataset_path}/lan.txt', 'r') as file:
            data['lang']['code'] = file.read()
    except:
        pass
       
    try:
        with open(f'{dataset_path}/language.txt', 'r') as file:
            data['lang']['name'] = file.read()  
    except:
        pass
        
    # Load info
    try:
        with open(f'{dataset_path}/README.txt', 'r') as file:
            data['info'] = file.read()
        
    except:
          pass

    try:
        with open(f'{dataset_path}/README', 'r') as file:
            data['info'] = file.read()
        
    except:
          pass
            
    # Load corpora
    filenames = os.listdir(f'{dataset_path}/docsutf8')
    corpora_ids = list(map(_remove_extension, filenames))
    data['corpora'] = []
    
    for corpus_id in corpora_ids:
        corpus = {}
        corpus['id'] = corpus_id

        try:
            with open(f'{dataset_path}/docsutf8/{corpus_id}.txt', 'r') as file:
                corpus['text'] = file.read()
                
            with open(f'{dataset_path}/keys/{corpus_id}.key', 'r') as file:
                corpus['keywords'] = file.read().split('\n')
                
            data['corpora'].append(corpus)
        except:
            pass
    
    return data

def _remove_extension(filename):
    return os.path.splitext(filename)[0]


def _url_of(dataset):
    return f'https://github.com/wilcoln/KeywordExtractor-Datasets/raw/master/datasets/{dataset}.zip'