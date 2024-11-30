import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import uuid


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


class InformerTrainer:
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
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-8)

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
                #loss = F.cross_entropy(outputs, y)
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
                '--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(), acc,
                                                                                               best_acc))
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


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Informer, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc):
        x_enc = torch.transpose(x_enc, 1, 2)
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):
        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]

