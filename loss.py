import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from pathlib import Path
from PIL import ImageFile
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter # logging outputs without slowing down training
import transformer
import stytr_just_loss as StyTR
import wandb

def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = self.transform(Image.open(str(path)).convert('RGB'))
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', default='content', type=str,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='style', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--final_dir', default='final', type=str,
                    help='Directory path to the styles images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_threads', type=int, default=10)
parser.add_argument('--cape', default=True, type=bool, choices=(True, False),
                        help="Use of Content-Aware Positional Encoding")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--num_gpus', default=1, type=int, help="Number of GPUs for training")
parser.add_argument('--activation_func', default='relu', help="Activation Function while training (ReLU, GeLU, GLU)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder_arch
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer(activation_func=args.activation_func, cape=args.cape)
with torch.no_grad():
    network = StyTR.StyTrans(vgg,decoder,embedding, Trans,args)

network.to(device)
content_tf = train_transform()
style_tf = train_transform()
final_tf = train_transform()



content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)
final_dataset = FlatFolderDataset(args.final_dir, final_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
final_iter = iter(data.DataLoader(
    final_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(final_dataset),
    num_workers=args.n_threads)) 

content_images = next(content_iter).to(device)
style_images = next(style_iter).to(device)
final_images = next(final_iter).to(device)

content_loss, style_loss = network(content_images, style_images, final_images)

print(f"Content Loss: {content_loss}")
print(f"Style Loss: {style_loss}")



