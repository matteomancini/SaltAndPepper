from torch.utils.data import Dataset
from util.transforms import get_transform


class Series(Dataset):
    def __init__(self, ts, labels, opt={}):
        self.ts = ts
        self.labels = labels
        self.n_series = ts.size(dim=1)
        self.t_len = ts.size(dim=2)
        self.n_labels = len(labels)
        self.transform = get_transform(opt)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        return self.transform(self.ts[index, :, :]), self.labels[index]

