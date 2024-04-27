import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.fc1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        if pos is not None:
            temp_tgt2 = tgt2 + query_pos
        else:
            temp_tgt2 = tgt2
        q = k = temp_tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt2, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.fc1(tgt2)
        tgt2 = self.relu(tgt2)
        tgt2 = self.fc2(tgt2)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt        


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                          pos=pos, query_pos=query_pos)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output.unsqueeze(0)
