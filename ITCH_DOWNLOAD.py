import warnings
warnings.filterwarnings('ignore')
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin

data_path = Path(r'clever-trade-bot-ai-main/P_project_with_python/Data_sources/GIT_DATASETS')

URL_ITCH = 'https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/'
FIELD_ICTH = 'tvagg.gz'

def may_be_download(url):
    """Download & unzip ITCH data if not yet available"""
    if not data_path.exists():
        print('Creating directory')
        data_path.mkdir()
    else: 
        print('Directory exists')

    filename = data_path / url.split('/')[-1]        
    if not filename.exists():
        print('Downloading...', url)
        urlretrieve(url, filename)
    else: 
        print('File exists')        

    unzipped = data_path / (filename.stem + '.bin')
    if not unzipped.exists():
        print('Unzipping to', unzipped)
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else: 
        print('File already unpacked')
    return unzipped

#field_name = may_be_download(urljoin(URL_ITCH, FIELD_ICTH))

