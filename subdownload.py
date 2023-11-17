import os
import shutil

path_data = Path('data')
path_data.mkdir(exist_ok=True)
path = path_data/'bedroom'

url = 'https://s3.amazonaws.com/fast-ai-imageclas/bedroom.tgz'
if not path.exists():
    path_zip = fc.urlsave(url, path_data)
    shutil.unpack_archive('data/bedroom.tgz', 'data')