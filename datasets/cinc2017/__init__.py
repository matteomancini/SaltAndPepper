import csv
import numpy as np
import os.path
import requests
import torch
import zipfile
from scipy.io import loadmat, savemat


def download_data():
    dataset = os.path.join(os.path.dirname(__file__), 'training2017.zip')
    if os.path.isfile(dataset):
        print(f"Dataset already downloaded in {os.path.dirname(__file__)}.")
    else:
        url = 'https://archive.physionet.org/challenge/2017/training2017.zip'
        r = requests.get(url, allow_redirects=True)
        with open(dataset, 'wb') as f:
            f.write(r.content)
    return dataset


def assemble_data():
    datamat = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if os.path.isfile(datamat):
        print(f"Dataset already assembled in {os.path.dirname(__file__)}.")
    else:
        labels_to_pick = {'N': 0, 'A': 1}
        data = []
        labels = []
        dataset = download_data()
        with zipfile.ZipFile(dataset, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(__file__))
        with open(os.path.join(os.path.dirname(__file__), 'training2017', 'REFERENCE.csv'), 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] in labels_to_pick.keys():
                    d = loadmat(os.path.join(os.path.dirname(__file__), 'training2017', ''.join([row[0],'.mat'])))
                    data.append(d['val'][0])
                    labels.append(labels_to_pick[row[1]])
        data = np.array(data, dtype=object)
        labels = np.array(labels)
        label_names = list(labels_to_pick.keys())
        savemat(datamat, {'data': data, 'labels': labels, 'label_names': label_names})


def load_data(t_len=9000):
    filename = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if ~os.path.isfile(filename):
        assemble_data()
    struct = loadmat(filename)
    data = struct['data'][0]
    samples = len(data)
    n_series = 1
    ts = np.zeros((samples, n_series, t_len), dtype=np.float32)
    for n in range(n_series):
        if len(data[n][0]) >= t_len:
            ts[n, 0, :] = data[n][0][:t_len]
        else:
            ts[n, 0, len(data[n][0])] = data[n][0]
    labels = struct['labels'][0]
    return torch.from_numpy(ts), torch.from_numpy(labels)


def get_label_names():
    filename = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if ~os.path.isfile(filename):
        assemble_data()
    struct = loadmat(filename)
    names = struct['label_names']
    return names
