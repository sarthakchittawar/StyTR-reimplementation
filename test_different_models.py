import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import transformer as transformer
import stytr as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time
from tqdm import tqdm


def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
# parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
# parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
# parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')
parser.add_argument('--decoder_path', type=str, default='/scratch/sanika/experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='/scratch/sanika/experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='/scratch/sanika/experiments/embedding_iter_160000.pth')


parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()




# Advanced options
content_size=256
style_size=256
crop='store_true'
save_ext='.jpg'
output_path=args.output
preserve_color='store_true'
alpha=args.a




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]    
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if not os.path.exists(output_path):
    os.mkdir(output_path)

for i in tqdm(range(10000, 170000, 10000)):
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder_arch
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(f'/scratch/sanika/experiments/decoder_iter_{i}.pth')
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(f'/scratch/sanika/experiments/transformer_iter_{i}.pth')
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(f'/scratch/sanika/experiments/embedding_iter_{i}.pth')
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
    network.eval()
    network.to(device)



    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    content_tf1 = content_transform()       
    content = content_tf(Image.open('content.png').convert("RGB"))

    h,w,c=np.shape(content)    
    style_tf1 = style_transform(h,w)
    style = style_tf(Image.open('style.png').convert("RGB"))


    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    
    with torch.no_grad():
        output= network(content,style)       
    output = output[0].cpu()
            
    output_name = f'{output_path}/output_{i}.jpg'

    save_image(output, output_name)
   