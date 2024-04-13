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
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if pos is not None:
            src_with_pos_embed = src + pos
        else:
            src_with_pos_embed = src
            
        q = k = src_with_pos_embed
        
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src += self.dropout(src2)
        src = nn.LayerNorm(self.embed_dim)(src)
        src2 = self.fc2(self.relu(self.fc1(src)))
        src += self.dropout(src2)
        src = nn.LayerNorm(self.embed_dim)(src)
        
        return src
        
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output