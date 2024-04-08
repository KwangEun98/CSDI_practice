import os
import sys
import argparse
import requests
import pickle
import tarfile
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str)
parser.add_argument('--dataset', type = str, default = 'physio', choices = ['physio', 'pm25'])
args = parser.parse_args()

def download_file(root_dir, dataset):
    save_dir = os.path.join(root_dir, 'dataset')
    os.makedirs(save_dir, exist_ok=True)

    if dataset == 'physio':
        url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
        content = requests.get(url).content
        with open(os.path.join(save_dir, f"{dataset}.tar.gz"), 'wb') as f:
            f.write(content)

        with tarfile.open(os.path.join(save_dir, f"{dataset}.tar.gz"), "r:gz") as tar:
            tar.extractall(save_dir)

    elif dataset == 'pm25':
        url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
        content = requests.get(url).content
        with open(os.path.join(save_dir, f"{dataset}.zip"), 'wb') as f:
            f.write(content)

        with zipfile.ZipFile(os.path.join(save_dir, f"{dataset}.zip")) as zip:
            zip.extractall(save_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

if __name__ == '__main__':
    download_file(args.root_dir, args.dataset)
