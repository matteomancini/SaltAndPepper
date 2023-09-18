from torch.utils.data import Dataset


class Series(Dataset):
    def __init__(self, ts, labels):
        self.ts = ts
        self.labels = labels
        self.n_series = ts.size(dim=1)
        self.t_len = ts.size(dim=2)
        self.n_labels = len(labels)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        return self.ts[index, :, :], self.labels[index]

