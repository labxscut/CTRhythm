import math
import torch
from torch import nn

class Residual1(nn.Module):
    def __init__(self, channelInput=32, channel=32, stride=2, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(channelInput, channel, kernel_size=3, padding=0, stride=stride)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(channel, channel, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(channelInput, channel, kernel_size=3, padding=0, stride=stride)
        self.norm1 = nn.BatchNorm1d(channel)
        self.norm2 = nn.BatchNorm1d(channel)
        self.norm3 = nn.BatchNorm1d(channel)
        self.ReLU = nn.ReLU()

    def forward(self, X):
        X = self.dropout(X)
        Y = self.ReLU(self.norm1(self.conv1(X)))
        Y = self.ReLU(self.norm2(self.conv2(Y)))
        Y = self.ReLU(self.norm3(self.conv3(Y)))
        X = self.conv4(X)
        return Y + X


class Residual2(nn.Module):
    def __init__(self, channelInput=32, channel=32, stride=1, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(channelInput, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(channel, channel, kernel_size=3, padding=1)

        self.norm1 = nn.BatchNorm1d(channel)
        self.norm2 = nn.BatchNorm1d(channel)
        self.norm3 = nn.BatchNorm1d(channel)
        self.ReLU = nn.ReLU()

    def forward(self, X):
        X = self.dropout(X)
        Y = self.ReLU(self.norm1(self.conv1(X)))
        Y = self.ReLU(self.norm2(self.conv2(Y)))
        Y = self.ReLU(self.norm3(self.conv3(Y)))
        return Y + X


class ResLayers(nn.Module):
    def __init__(self, add_depth, dropout=0.3):
        super().__init__()
        self.main = nn.Sequential(
            Residual1(1, 16, 2, dropout),
            Residual1(16, 32, 2, dropout),
            Residual1(32, 32, 2, dropout),
            Residual1(32, 32, 2, dropout),
            Residual1(32, 32, 2, dropout)
        )
        for _ in range(add_depth):
            self.main.append(
                Residual2(32, 32, 1, dropout)
            )

    def forward(self, x):
        x = self.main(x)
        return x



class TransformerEncoder(nn.Module):
    def __init__(self,d_in,d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_in,d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 4)  # num_class

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.pos_encoding(x)  # [seq_len, batch_size, d_model]
        x = self.transformer_encoder(x)  # [seq_len, batch_size, d_model]
        x = torch.mean(x, dim=0)  # [batch_size, d_model]
        x = self.fc(x)  # [batch_size, num_classes]
        return x

class CTRhythm(nn.Module):
    def __init__(self, d_in,d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(
            ResLayers(1, 0.3),
            TransformerEncoder(d_in,d_model, nhead, num_layers, dim_feedforward, dropout)
        )
    def forward(self, x):
        x = self.main(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_in,d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.linear = nn.Linear(d_in, d_model)

    def forward(self, x):
        x = self.linear(x)  # Apply linear layer
        x = x + self.pe[:x.size(0), :]  # Add positional encoding
        x = self.dropout(x)  # Apply dropout
        return x
