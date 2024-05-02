'''
Encoder module for the transformer model
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from transformer import get_activation_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One layer of the encoder
class EncoderLayer(nn.Module):
    ''' Class representing one encoder layer '''
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout, activation_func='relu'):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.fc1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_dim)
        
        self.actv = get_activation_func(activation_func)
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    # q: query, k: key, v: value
    def forward(self, source, source_mask=None, source_key_padding_mask=None, position=None):
        # if positional encodings are provided, they are added to the source
        if position is not None:
            source_with_pos_embed = source + position
        else:
            source_with_pos_embed = source

        # self attention: the source with positional encodings is used as the query and the key 
        q = k = source_with_pos_embed
        
        # this part takes the query and key as input and computes the attention weights
        # the index [0] is used to get the output of the multihead attention layer
        source2 = self.self_attn(q, k, value=source, attn_mask=source_mask, key_padding_mask=source_key_padding_mask)[0]
        
        # residual connection and layer normalization
        source = source + self.dropout(source2)
        source = self.norm1(source)
        source2 = self.fc2(self.actv(self.fc1(source)))
        source = source + self.dropout(source2)
        source = self.norm2(source)
        
        return source
        
class Encoder(nn.Module):
    ''' Class representing the Encoder module '''
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # initialize the layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, source, mask=None, source_key_padding_mask=None, position=None):
        # for each layer, the output is passed as the input to the next layer
        # the output is normalized at the end
        for layer in self.layers:
            output = layer(source, source_mask=mask, source_key_padding_mask=source_key_padding_mask, position=position)
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output