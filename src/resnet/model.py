import os
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np


class ResNetTrainer:
    def __init__(self, device, logger, DP=-1):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.DP = DP

    def fit(self, model, X_train, y_train, epochs=100, batch_size=128, eval_batch_size=128):
        print('Training model for {} epochs'.format(epochs))
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        train_dataset = Dataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-3)
        if self.DP > 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=4e-3, eps=self.DP)

        best_acc = 0.0
        patient = 70
        best_loss = 1e10

        # Training
        for epoch in range(epochs):
            model.train()
            for i, inputs in enumerate(train_loader, 0):
                X, y = inputs
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(X)
                if len(y.size()) == 1:
                    loss = F.nll_loss(outputs, y)
                else:
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
                patient = 70
            else:
                patient -= 1
                if patient == 0:
                    break
            try:
                self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc))
            except:
                print('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc))
        
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


class ResNet(nn.Module):

    def __init__(self, input_size, nb_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 64

        self.block_1 = ResNetBlock(input_size, n_feature_maps)
        self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.linear = nn.Linear(n_feature_maps, nb_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = F.avg_pool1d(x, x.shape[-1]).view(x.shape[0],-1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.expand = True if in_channels < out_channels else False

        self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn_z = nn.BatchNorm1d(out_channels)

        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        if self.expand:
            x = self.shortcut_y(x)
        x = self.bn_shortcut_y(x)
        out += x
        out = F.relu(out)

        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


