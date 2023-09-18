from data import series_data
from importlib import import_module
from util import train
from torch.utils.data import DataLoader, random_split

dataset_name = 'cinc2017'
model_name = 'FullyConnectedNetwork'
units = 100
layers = 2
batches = 100
epochs = 20

dataset = import_module('datasets.' + dataset_name)
model = getattr(import_module('model'), model_name)
ts, labels = dataset.load_data()
data = series_data.Series(ts, labels)
train_size = int(0.8 * len(data))
valid_size = len(data) - train_size
train_data, valid_data = random_split(data, [train_size, valid_size])
train_loader = DataLoader(train_data, batch_size=batches, shuffle=True)
validation_loader = DataLoader(valid_data, batch_size=batches, shuffle=True)
net = model(num_classes=labels.max()+1, input_len=ts.size(2),
                            hidden_units=units, hidden_layers=layers).to('cpu')
print(f'The number of samples for training is {train_size}.')
print(f'The number of parameters is {sum(p.numel() for p in net.parameters())}.')
train_loss, train_acc, validation_loss, validation_acc, = train(net, 'cpu', train_loader, validation_loader, epochs)
