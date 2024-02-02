import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, device, train_loader, validation_loader, epochs,
          lr=0.01, weight_decay=0, loss_fn=nn.CrossEntropyLoss(), no_labels=False):
    criterion = loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []
    with tqdm(range(epochs), unit='epoch') as tepochs:
        tepochs.set_description('Training')
        for epoch in tepochs:
            model.train()
            running_loss = 0
            correct, total = 0, 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                optimizer.zero_grad()
                if not no_labels:
                    loss = criterion(output, target)
                else:
                    loss = criterion(output, output)
                loss.backward()
                optimizer.step()
                tepochs.set_postfix(loss=loss.item())
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            train_loss.append(running_loss / len(train_loader))
            train_acc.append(correct / total)

            model.eval()
            running_loss = 0
            correct, total = 0, 0
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                if not no_labels:
                    loss = criterion(output, target)
                    tepochs.set_postfix(loss=loss.item())
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == target).sum().item()
                else:
                    loss = criterion(output, output)
                running_loss += loss.item()
                total += target.size(0)
            validation_loss.append(running_loss / len(validation_loader))
            if not no_labels:
                validation_acc.append(correct / total)

        target_list, predicted_list = [], []
        if not no_labels:
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                _, predicted = torch.max(output, 1)
                predicted_list.extend(predicted)
                target_list.extend(target)

    return train_loss, train_acc, validation_loss, validation_acc, predicted_list, target_list
