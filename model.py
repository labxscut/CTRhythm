import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel, stride = 1):
        super().__init__()
        self.conv = nn.Conv1d(channel_in, channel, kernel_size=3, padding=1, stride=stride)
        self.norm = nn.BatchNorm1d(channel)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)   
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channel_in = 32, channel = 32, stride = 1, dropout = 0.3):
        super().__init__()
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.main = nn.Sequential(
            ConvBlock(channel_in, channel, stride = stride),
            ConvBlock(channel, channel),
            ConvBlock(channel, channel),
        )
        self.skip = None
        if self.stride == 2:
            self.skip = nn.Conv1d(channel_in, channel, kernel_size = 3, padding = 1, stride=stride)
  
    def forward(self, x):
        x = self.dropout(x)
        y = self.main(x)
        if self.stride == 2:
            x = self.skip(x)
        x = y + x   
        return x
class CNNModule(nn.Module):
    def __init__(self, channel_in = 1,downsample_depth = 5, main_depth = 1 ,dropout=0.3):
        super().__init__()
        
        self.downsample_layers = nn.Sequential(
            ResidualBlock(channel_in, 16, 2, dropout),
            ResidualBlock(16, 32, 2, dropout),
        )
        self.main = nn.Sequential(
            ResidualBlock(dropout = dropout)
        )
        if downsample_depth > 2:
            for i in range(downsample_depth - 2):
                self.downsample_layers.append(ResidualBlock(32,32,2,dropout=dropout))
        if main_depth > 1:
            for i in range(main_depth - 1):
                self.downsample_layers.append(ResidualBlock(32,32,1,dropout=dropout))

    def forward(self, x):
        x = self.downsample_layers(x)
        x = self.main(x) # [batch_size, d_model, seq_len]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_in,d_model, dropout=0.1, max_len=10000):
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

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)  # [seq_len, batch_size, d_model]
        return x

class CTRhythm(nn.Module):
    def __init__(self, channel_in = 1, d_model = 32, nhead=8, num_layers = 6, dim_feedforward = 256, num_classes = 4, dropout=0.1):
        super().__init__()
        self.CNN = CNNModule(channel_in = channel_in, downsample_depth = 5, main_depth = 1 ,dropout = 0.3)
        self.pos_encoding = PositionalEncoding(32, d_model, dropout)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout = dropout)
        self.classification = nn.Linear(d_model, num_classes)    
    def forward(self, x):
        x = self.CNN(x) # [batch_size, d_model, seq_len]
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x) # [seq_len, batch_size, d_model]
        x = self.transformer_encoder(x) # [seq_len, batch_size, d_model]
        x = torch.mean(x, dim=0)  # [batch_size, d_model]
        x = self.classification(x)  # [batch_size, num_classes]
        return x
