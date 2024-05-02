'''
The Style transformer module
'''

import torch.nn.functional as F
import torch.nn as nn

def get_activation_func(actv):
    '''Return the activation function given a string'''
    if actv == 'relu':
        return F.relu
    if actv == 'gelu':
        return F.gelu
    if actv == 'glu':
        return F.glu
    raise RuntimeError(F"Activation Function should be relu/gelu/glu, not {actv}!")

from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
import numpy as np

class Transformer(nn.Module):
    '''The Style transformer consisting of 2 encoders - one for the content & style images each (w/ multiple layers) and a decoder.
       The architecture uses Layer Normalisation and dropout layers for training to help stabilize the network and also reduce effects of noise.
    '''
    def __init__(self, dimensions=512, num_heads=8, num_encoder_layers=3, 
                 num_decoder_layers=3, feedforward_dimensions=2048, dropout=0.1, activation_func='relu'):
        
        super(Transformer, self).__init__()
        self.dimensions = dimensions
        self.num_heads = num_heads
        
        # encoders
        encoder_layer = EncoderLayer(dimensions, num_heads, feedforward_dimensions, dropout, activation_func)
        self.encoder_style = Encoder(encoder_layer, num_encoder_layers)
        self.encoder_content = Encoder(encoder_layer, num_encoder_layers)
        
        # decoder
        decoder_layer = DecoderLayer(dimensions, num_heads, feedforward_dimensions, dropout, activation_func)
        decoder_norm = nn.LayerNorm(dimensions)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        # Initialise weights using Xavier uniform initialisation method
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # apply linear transformations to each element in the feature map independently of its neighbors
        self.conv = nn.Conv2d(dimensions, dimensions, (1, 1))
        
        # to ensure fixed size output
        self.average_pool = nn.AdaptiveAvgPool2d(18)
        
    def forward(self, style, content, pos_encoding_style, pos_encoding_content, mask, cape=True):
        
        if cape:
            # Get the CAPE encoding
            positional_content = self.conv(self.average_pool(content))
            pos_encoding_content = F.interpolate(positional_content, size=style.shape[-2:], mode='bilinear')
        
        # Flatten and permute style and content tensors for compatibility with Transformer layers
        style = style.flatten(2).permute(2, 0, 1)
        if pos_encoding_style is not None:
            pos_encoding_style = pos_encoding_style.flatten(2).permute(2, 0, 1)
        content = content.flatten(2).permute(2, 0, 1)
        if pos_encoding_content is not None:
            pos_encoding_content = pos_encoding_content.flatten(2).permute(2, 0, 1)

        # Pass style and content through encoder layers
        style = self.encoder_style(style, mask, pos_encoding_style)
        content = self.encoder_content(content, mask, pos_encoding_content)
        
        # Pass encoded style and content through decoder
        output = self.decoder(content, style, mask)[0]
        
        # rearranging dimensions for viewing in batches
        N, B, C= output.shape       
        H = int(np.sqrt(N))
        output = output.permute(1, 2, 0)
        output = output.view(B, C, -1, H)
        
        return output