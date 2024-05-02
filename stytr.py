import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# from torch._six import container_abcs
import collections.abc as container_abcs # above import statement is deprecated
from itertools import repeat
from typing import Optional, List
from torch import Tensor

# consists of the decoder architecture
decoder_arch = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# consists of the extended encoder backbone architecture out of which a subset is used (i.e. until relu4-1)
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def make_tuple(x):
    ''' Makes a tuple out of an iterable while repeating its elements twice '''
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

class NestedTensor(object):
    ''' TODO: define this '''
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    ''' TODO: define this '''
    if tensor_list[0].ndim == 3:        
        l = [list(img.shape) for img in tensor_list]
        max_size = l[0]
        for sublist in l[1:]:
            for index, item in enumerate(sublist):
                max_size[index] = max(max_size[index], item)
        
        batch_shape = [len(tensor_list)] + max_size

        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class PatchEmbed(nn.Module):
    ''' TODO: define this '''
    def __init__(self, img_size=256, patch_size=8, in_channels=3, embed_dim=512):
        super().__init__()
        img_size = make_tuple(img_size)
        patch_size = make_tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        x = self.proj(x)
        return x
    
class StyTrans(nn.Module):
    """ Full Style Transfer Module """
    def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for enc in [self.enc_1, self.enc_2, self.enc_3, self.enc_4, self.enc_5]:
            for param in enc.parameters():
                param.requires_grad = False
                
        self.transformer = transformer
        hidden_dim = transformer.dimensions
        self.mse_loss = nn.MSELoss()
        self.decoder = decoder
        self.embedding = PatchEmbed
        
    def intermediate_encoding(self, x):
        """ Get intermediate encodings """
        results = [x]
        for module in [self.enc_1, self.enc_2, self.enc_3, self.enc_4, self.enc_5]:
            x = module(results[-1])
            results.append(x)
        return results[1:]
    
    def calc_content_loss(self, content_input, content_target):
        ''' Function to calculate content loss while training '''
        assert content_input.size() == content_target.size()
        assert ~content_target.requires_grad
        return self.mse_loss(content_input, content_target)
    
    def calc_mean_std(self, vec):
        ''' Function to calculate mean & variance of a vector '''
        size = vec.size()
        assert (len(size) == 4)
        
        N, C = size[:2]
        variance = vec.view(N, C, -1).var(dim=2) + 1e-5 # epsilon = 1e-5
        std = variance.sqrt().view(N, C, 1, 1)
        mean = vec.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return mean, std
    
    def calc_style_loss(self, style_input, style_target):
        ''' Function to calculate style loss while training '''
        assert style_input.size() == style_target.size()
        assert ~style_target.requires_grad
        
        # return effective loss as sum of mean mse and std mse
        input_mean, input_std = self.calc_mean_std(style_input)
        target_mean, target_std = self.calc_mean_std(style_target)
        
        eff_loss = self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)
        return eff_loss
        
        
    def forward(self, content: NestedTensor, style: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        content_init = content
        style_init = style
        
        if isinstance(content, (list, Tensor)):
            content = nested_tensor_from_tensor_list(content)
        if isinstance(style, (list, Tensor)):
            style = nested_tensor_from_tensor_list(style)
        
        content_features = self.intermediate_encoding(content.tensors)
        style_features = self.intermediate_encoding(style.tensors)
        
        content_proj = self.embedding(content.tensors)
        style_proj = self.embedding(style.tensors)
        
        # postional embedding calculated in transformer.py
        pos_c = None
        pos_s = None
        mask = None
        
        # forward pass through the transformer
        trans = self.transformer(style_proj, content_proj, pos_s, pos_c, mask)
        
        cs = self.decoder(trans)
        trans_features = self.intermediate_encoding(cs)
        
        # content loss
        content_input = trans_features[-1]
        m, s = self.calc_mean_std(content_input)
        normalised_content_input = (content_input - m) / s
        
        content_target = content_features[-1]
        m, s = self.calc_mean_std(content_target)
        normalised_content_target = (content_target - m) / s
        
        content_loss = self.calc_content_loss(normalised_content_input, normalised_content_target)
        
        content_input = trans_features[-2]
        m, s = self.calc_mean_std(content_input)
        normalised_content_input = (content_input - m) / s
        
        content_target = content_features[-2]
        m, s = self.calc_mean_std(content_target)
        normalised_content_target = (content_target - m) / s
        
        content_loss += self.calc_content_loss(normalised_content_input, normalised_content_target)
        
        # style loss
        style_loss = 0
        for i in range(5):
            style_loss += self.calc_style_loss(trans_features[i], style_features[i])            
        
        # Identity loss
        cc = self.decoder(self.transformer(content_proj, content_proj, pos_c, pos_c, mask))
        ss = self.decoder(self.transformer(style_proj, style_proj, pos_s, pos_s, mask))
        
        i_loss1 = self.calc_content_loss(cc, content_init) + self.calc_content_loss(ss, style_init)
        
        i_loss2 = 0
        cc_features = self.intermediate_encoding(cc)
        ss_features = self.intermediate_encoding(ss)
        for i in range(5):
            i_loss2 += self.calc_style_loss(cc_features[i], content_features[i]) + self.calc_content_loss(ss_features[i], style_features[i])
            
        return cs, content_loss, style_loss, i_loss1, i_loss2
