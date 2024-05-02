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
    ''' Decodes the encoded content/style embeddings to a combined content embedding with style transfer '''
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout, activation_func='relu'):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
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
        
    def forward(self, enc_content, enc_style, content_mask=None, style_mask=None, content_key_padding_mask=None, style_key_padding_mask=None, style_pos=None, content_pos=None):        
        if content_pos is not None:
            query = enc_content + content_pos
        else:
            query = enc_content
        
        if style_pos is not None:
            key = enc_style + style_pos
        else:
            key = enc_style
        value = enc_style
        
        # query: encoded content embedding, key & value: encoded style embedding -- for self-attention
        output = self.self_attn(query, key, value, attn_mask=content_mask, key_padding_mask=content_key_padding_mask)[0]
        enc_content = enc_content + self.dropout1(output)
        enc_content = self.norm1(enc_content)
        
        if content_pos is not None:
            query = enc_content + content_pos
        else:
            query = enc_content
        
        # query: encoded content embedding with added dropout, key & value: encoded style embedding -- for multihead-attention
        output = self.multihead_attn(query=query, key=key, value=value, attn_mask=style_mask, key_padding_mask=style_key_padding_mask)[0]
        enc_content = enc_content + self.dropout2(output)
        enc_content = self.norm2(enc_content)
        
        # fully connected layers + dropout layers
        output = self.fc2(self.dropout(self.actv(self.fc1(enc_content))))
        enc_content = enc_content + self.dropout3(output)
        enc_content = self.norm3(enc_content)
        
        return enc_content

class Decoder(nn.Module):
    ''' Class representing the Decoder module '''
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, enc_content, enc_style, content_mask=None, style_mask=None, content_key_padding_mask=None, style_key_padding_mask=None, style_pos=None, content_pos=None):
        output = enc_content
        for layer in self.layers:
            output = layer(output, enc_style, content_mask=content_mask, style_mask=style_mask,
                          content_key_padding_mask=content_key_padding_mask, style_key_padding_mask=style_key_padding_mask,
                          style_pos=style_pos, content_pos=content_pos)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output.unsqueeze(0)
