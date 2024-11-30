import os
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def compute_accuracy(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            X, labels = data
            X, labels = X.to(device), labels.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if len(labels.size()) == 2:
                correct += (predicted == labels.argmax(axis=1)).sum().item()
            else:
                correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc


def get_proba(model, loader, device):

    output_list = []
    with torch.no_grad():
        for data in loader:
            X, labels = data
            X, labels = X.to(device), labels.to(device)
            outputs = F.softmax(model(X), dim=1)
            output_list.append(outputs.cpu().numpy())
    outputs_all = np.concatenate(output_list)
    return outputs_all


class ConvNetTrainer:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def fit(self, model, X_train, y_train, epochs=100, batch_size=128, eval_batch_size=128):
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        train_dataset = Dataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-3)

        best_acc = 0.0
        patient = 10
        best_loss = 1e10

        # Training
        for epoch in range(epochs):
            model.train()
            for i, inputs in enumerate(train_loader, 0):
                X, y = inputs
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(X)
                #loss = F.nll_loss(outputs, y)
                loss = F.multilabel_soft_margin_loss(outputs, y)
                loss.backward()
                optimizer.step()

            model.eval()
            acc = compute_accuracy(model, train_loader, self.device)
            '''if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), file_path)'''
            if loss.item() < best_loss - 0.0002:
                best_acc = acc
                best_loss = loss.item()
                torch.save(model.state_dict(), file_path)
                patient = 10
            else:
                patient -= 1
                if patient == 0:
                    break
            try:
                self.logger.log(
                    '--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(),
                                                                                                   acc, best_acc))
            except:
                print('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(),
                                                                                                     acc, best_acc))

        # Load the best model
        model.load_state_dict(torch.load(file_path))
        model.eval()
        os.remove(file_path)

        return model

    def test(self, model, X_test, y_test, batch_size=128):
        test_dataset = Dataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        acc = compute_accuracy(model, test_loader, self.device)
        return acc

    def pred_proba(self, model, X_test, y_test, batch_size=128):
        test_dataset = Dataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        probs = get_proba(model, test_loader, self.device)
        return probs


class ConvNet(nn.Module):
    def __init__(self,n_in, n_classes, n_ft):
        super(ConvNet,self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes
        self.n_ft = n_ft

        self.conv1 = nn.Conv2d(self.n_ft,128,(7,1),1,(3,0))
        self.bn1   = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128,256,(5,1),1,(2,0))
        self.bn2   = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256,128,(3,1),1,(1,0))
        self.bn3   = nn.BatchNorm2d(128)

        self.fc4   = nn.Linear(128,self.n_classes)


    def forward(self, x: torch.Tensor):
        x = x.view(-1,self.n_ft,self.n_in,1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        #x = F.avg_pool2d(x.view(x.shape[:-1]),2)
        x = torch.mean(x, dim=2)
        x = x.view(-1,128)
        x = self.fc4(x)

        return F.log_softmax(x,1)

