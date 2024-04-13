import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.fc1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_dim)
        
        self.relu = nn.ReLU()
        self.embed_dim = embed_dim
        
    # TODO: Implement the `forward` method for the `EncoderLayer` class
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        pass
        
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return output