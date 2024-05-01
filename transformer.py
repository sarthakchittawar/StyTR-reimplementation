import torch.nn.functional as F
import torch.nn as nn
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dimensions=512, num_heads=8, num_encoder_layers=3, 
                 num_decoder_layers=3, feedforward_dimensions=2048, dropout=0.1):
        
        super(Transformer, self).__init__()
        self.dimensions = dimensions
        self.num_heads = num_heads
        encoder_layer = EncoderLayer(dimensions, num_heads, feedforward_dimensions, dropout)
        self.encoder_style = Encoder(encoder_layer, num_encoder_layers)
        self.encoder_content = Encoder(encoder_layer, num_encoder_layers)
        decoder_layer = DecoderLayer(dimensions, num_heads, feedforward_dimensions, dropout)
        decoder_norm = nn.LayerNorm(dimensions)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.conv = nn.Conv2d(dimensions, dimensions, (1, 1))
        self.average_pool = nn.AdaptiveAvgPool2d(18)
    def forward(self, style, content, pos_encoding_style, pos_encoding_content, mask):
        positional_content = self.conv(self.average_pool(content))
        positional_content = F.interpolate(positional_content, size=style.shape[-2:], mode='bilinear')
        style = style.flatten(2).permute(2, 0, 1)
        if pos_encoding_style is not None:
            pos_encoding_style = pos_encoding_style.flatten(2).permute(2, 0, 1)
        content = content.flatten(2).permute(2, 0, 1)
        if pos_encoding_content is not None:
            pos_encoding_content = pos_encoding_content.flatten(2).permute(2, 0, 1)

        style = self.encoder_style(style, mask, pos_encoding_style)
        content = self.encoder_content(content, mask, pos_encoding_content)
        output = self.decoder(content, style, mask)[0]
        N, B, C= output.shape       
        H = int(np.sqrt(N))
        output = output.permute(1, 2, 0)
        output = output.view(B, C, -1, H)
        return output

