import io
import os
import urllib.request
import zipfile
from collections import defaultdict
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'download')

def table():
    return pd.read_csv(io.StringIO(DATASETS))

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



DATASETS = """
Dataset,Language,Type of Doc,Domain,#Docs,#Gold Keys (per doc),#Tokens per doc,Absent GoldKey
110-PT-BN-KP,PT,News,Misc.,110,2610 (23.73),304.00,2.5%
500N-KPCrowd-v1.1,EN,News,Misc.,500,24459 (48.92),408.33,13.5%
Inspec,EN,Abstract,Comp. Science,2000,29230 (14.62),128.20,37.7%
Krapivin2009,EN,Paper,Comp. Science,2304,14599 (6.34),8040.74,15.3%
Nguyen2007,EN,Paper,Comp. Science,209,2369 (11.33),5201.09,17.8%
PubMed,EN,Paper,Comp. Science,500,7620 (15.24),3992.78,60.2%
Schutz2008,EN,Paper,Comp. Science,1231,55013 (44.69),3901.31,13.6%
SemEval2010,EN,Paper,Comp. Science,243,4002 (16.47),8332.34,11.3%
SemEval2017,EN,Paragraph,Misc.,493,8969 (18.19),178.22,0.0%
WikiNews,FR,News,Misc.,100,1177 (11.77),293.52,5.0%
cacic,ES,Paper,Comp. Science,888,4282 (4.82),3985.84,2.2%
citeulike180,EN,Paper,Misc.,183,3370 (18.42),4796.08,32.2%
fao30,EN,Paper,Agriculture,30,997 (33.23),4777.70,41.7%
fao780,EN,Paper,Agriculture,779,6990 (8.97),4971.79,36.1%
kdd,EN,Paper,Comp. Science,755,3831 (5.07),75.97,53.2%
pak2018,PL,Abstract,Misc.,50,232 (4.64),97.36,64.7%
theses100,EN,Msc/Phd Thesis,Misc.,100,767 (7.67),4728.86,47.6%
wicc,ES,Paper,Comp. Science,1640,7498 (4.57),1955.56,2.7%
wiki20,EN,Research Report,Comp. Science,20,730 (36.50),6177.65,51.8%
www,EN,Paper,Comp. Science,1330,7711 (5.80),84.08,55.0%
""" 