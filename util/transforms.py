import torch
import numpy as np
from importlib import import_module
from torchvision.transforms import Compose


def get_transform(opt):
    t_list = []
    if not opt.keys():
        return torch.nn.Identity()
    for t in opt.keys():
        t_type = getattr(import_module('util.transforms'), t)
        t_list.append(t_type(*opt[t]))
    return Compose(t_list)


class RandomAmplitudeFlip(torch.nn.Module):
    def __init__(self, p=0.5, f=1):
        super(RandomAmplitudeFlip, self).__init__()
        self.p = p
        self.f = f

    def forward(self, data):
        if torch.rand(1) < self.p:
            n_ts = len(data)
            selection = np.random.choice(n_ts, self.f, replace=False)
            data_to_flip = data[selection, :]
            data[selection, :] = -data_to_flip
            return data
        return data


class RandomScaling(torch.nn.Module):
    def __init__(self, p=0.5, lb=0.5, hb=1, f=1):
        super(RandomScaling, self).__init__()
        self.p = p
        self.lb = lb
        self.hb = hb
        self.f = f

    def forward(self, data):
        if torch.rand(1) < self.p:
            n_ts = len(data)
            selection = np.random.choice(n_ts, self.f, replace=False)
            data_to_amplify = data[selection, :]
            factor = (self.hb - self.lb) * np.random.random_sample() + self.lb
            data[selection, :] = factor * data_to_amplify
            return data
        return data


class RandomBiasShift(torch.nn.Module):
    def __init__(self, p=0.5, lb=0.01, hb=0.1, f=1):
        super(RandomBiasShift, self).__init__()
        self.p = p
        self.f = f
        self.lb = lb
        self.hb = hb

    def forward(self, data):
        if torch.rand(1) < self.p:
            n_ts = len(data)
            selection = np.random.choice(n_ts, self.f, replace=False)
            data_to_shift = data[selection, :]
            bias = (self.hb - self.lb) * np.random.random_sample() + self.lb
            data[selection, :] = bias + data_to_shift
            return data
        return data


class RandomOscillation(torch.nn.Module):
    def __init__(self, p=0.5, freq=0.5, lp=0.1, hp=2 * np.pi, amplitude=0.05, f_sample=250, f=1):
        super(RandomOscillation, self).__init__()
        self.p = p
        self.f = f
        self.freq = freq
        self.hp = hp
        self.lp = lp
        self.amplitude = amplitude
        self.f_sample = f_sample

    def forward(self, data):
        if torch.rand(1) < self.p:
            n_ts = len(data)
            selection = np.random.choice(n_ts, self.f, replace=False)
            data_to_add = data[selection, :]
            phase = (self.hp - self.lp) * np.random.random_sample() + self.lp
            t = np.linspace(0, data.shape[1] / self.f_sample, data.shape[1])
            oscillation = self.amplitude * torch.sin(2 * np.pi * self.freq * torch.from_numpy(t) + phase)
            oscillating_data = oscillation + data_to_add
            data[selection, :] = oscillating_data.float()
            return data
        return data


class MinMaxNormalization(torch.nn.Module):
    def __init__(self):
        super(MinMaxNormalization, self).__init__()

    def forward(self, data):
        data_min = torch.min(data, dim=1)[0]
        data_max = torch.max(data, dim=1)[0]
        data = (data - data_min) / (data_max - data_min)

        return data


class MeanStdNormalization(torch.nn.Module):
    def __init__(self):
        super(MeanStdNormalization, self).__init__()

    def forward(self, data):
        mean_data = data.mean(dim=1, keepdim=True)
        std_data = data.std(dim=1, keepdim=True)
        data = (data - mean_data) / std_data
        return data


class WhiteNoise(torch.nn.Module):
    def __init__(self, p=0.5, num_samples=1000, amplitude=0.5, f=1):
        super(WhiteNoise, self).__init__()
        self.p = p
        self.f = f
        self.amplitude = amplitude
        self.num_samples = num_samples

    def forward(self, data):
        if torch.rand(1) < self.p:
            n_ts = len(data)
            selection = np.random.choice(n_ts, self.f, replace=False)
            data_to_add = data[selection, :]
            mean = 0
            std = 1
            samples = np.random.normal(mean, std, self.num_samples)
            oscillating_data = data_to_add + self.amplitude * samples
            data[selection, :] = oscillating_data.float()
            return data
        return data
