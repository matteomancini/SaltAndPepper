from data import series_data
from importlib import import_module
from util import train
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

dataset_name = 'cinc2017'
model_name = 'FullyConnectedNetwork'
train_ae = False
rearrange_tensor = False
units = 150
layers = 1
batch = 150
epochs = 10
device = 'cpu'

dataset = import_module('datasets.' + dataset_name)
model = getattr(import_module('model'), model_name)
ts, labels = dataset.load_data()
if rearrange_tensor:
    ts = ts.transpose(1, 2)
n_labels = int(labels.max()) + 1
data = series_data.Series(ts, labels)
train_size = int(0.8 * len(data))
valid_size = len(data) - train_size
train_data, valid_data = random_split(data, [train_size, valid_size])
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
validation_loader = DataLoader(valid_data, batch_size=batch, shuffle=True)
net = model(num_classes=n_labels, input_len=ts.size(2),
            hidden_units=units, hidden_layers=layers).to(device)
print(f'The number of samples for training is {train_size}.')
print(f'The number of parameters is {sum(p.numel() for p in net.parameters())}.')
class_weights = [1/sum([int(t[1])==l for t in list(train_data)]) for l in range(n_labels)]
sample_weights = [class_weights[int(t[1])] for t in list(train_data)]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data), replacement=True)
rebalanced_loader = DataLoader(train_data, sampler=sampler, batch_size=batch)
train_loss, train_acc, validation_loss, validation_acc, predicted_list, target_list = train(
    net, device, rebalanced_loader, validation_loader, epochs, lr=0.01, no_labels=train_ae)
