import numpy as np
import os.path
import requests
import torch
import zipfile
from scipy.io import loadmat


def download_data():
    dataset = os.path.join(os.path.dirname(__file__), 'ECGData.zip')
    if os.path.isfile(dataset):
        print(f"Dataset already downloaded in {os.path.dirname(__file__)}.")
    else:
        url = 'https://github.com/mathworks/physionet_ECG_data/raw/main/ECGData.zip'
        r = requests.get(url, allow_redirects=True)
        with open(dataset, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(dataset, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(__file__))
    return dataset


def load_data():
    filename = os.path.join(os.path.dirname(__file__), 'ECGData.mat')
    if ~os.path.isfile(filename):
        download_data()
    struct = loadmat(filename)
    ts = np.float32(struct['ECGData']['Data'][0][0])
    samples, t_len = ts.shape
    n_series = 1
    ts = np.reshape(ts, (samples, n_series, t_len))

    labels = struct['ECGData']['Labels'][0][0]
    labels = [l[0][0] for l in labels]
    labels_unique = np.unique(labels)
    mapping = {l: n for n, l in enumerate(labels_unique)}
    labels = np.array([mapping[l] for l in labels])

    return torch.from_numpy(ts), torch.from_numpy(labels)


def get_label_names():
    filename = os.path.join(os.path.dirname(__file__), 'ECGData.mat')
    if ~os.path.isfile(filename):
        download_data()
    struct = loadmat(filename)
    labels = struct['ECGData']['Labels'][0][0]
    labels = [l[0][0] for l in labels]
    names = np.unique(labels)
    return names