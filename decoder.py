'''
Decoder module for the transformer model
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from transformer import get_activation_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One layer of the decoder
class DecoderLayer(nn.Module):
    ''' Class representing one decoder layer '''
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout, activation_func='relu'):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.fc1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.actv = get_activation_func(activation_func)
        self.embed_dim = embed_dim
        
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
    # q: query, k: key, v: value
    def forward(self, target, memory, target_mask=None, memory_mask=None, target_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):        
        if query_pos is not None:
            q = target + query_pos
        else:
            q = target
        
        if pos is not None:
            k = memory + pos
        else:
            k = memory
        v = memory
        
        tgt2 = self.self_attn(q, k, v, attn_mask=target_mask, key_padding_mask=target_key_padding_mask)[0]
        target = target + self.dropout1(tgt2)
        target = self.norm1(target)
        
        if query_pos is not None:
            q = target + query_pos
        else:
            q = target
        
        tgt2 = self.multihead_attn(query=q, key=k, value=v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        target = target + self.dropout2(tgt2)
        target = self.norm2(target)
        tgt2 = self.fc2(self.dropout(self.actv(self.fc1(target))))
        target = target + self.dropout3(tgt2)
        target = self.norm3(target)
        
        return target

class Decoder(nn.Module):
    ''' Class representing the Decoder module '''
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, target, memory, target_mask=None, memory_mask=None, target_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        output = target
        for layer in self.layers:
            output = layer(output, memory, target_mask=target_mask, memory_mask=memory_mask,
                          target_key_padding_mask=target_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                          pos=pos, query_pos=query_pos)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output.unsqueeze(0)
