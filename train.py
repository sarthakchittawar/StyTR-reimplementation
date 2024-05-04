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
import stytr as StyTR
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

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', default='/scratch/sarthak/datasets/train2017', type=str,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='/scratch/sarthak/datasets/style', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')

parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=10)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--cape', default=True, type=bool, choices=(True, False),
                        help="Use of Content-Aware Positional Encoding")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--num_gpus', default=1, type=int, help="Number of GPUs for training")
parser.add_argument('--activation_func', default='relu', help="Activation Function while training (ReLU, GeLU, GLU)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder_arch
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer(activation_func=args.activation_func, cape=args.cape)
with torch.no_grad():
    network = StyTR.StyTrans(vgg,decoder,embedding, Trans,args)
network.train()

network.to(device)
network = nn.DataParallel(network, device_ids=[i for i in range(args.num_gpus)])
content_tf = train_transform()
style_tf = train_transform()



content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
 

optimizer = torch.optim.Adam([ 
                              {'params': network.module.transformer.parameters()},
                              {'params': network.module.decoder.parameters()},
                              {'params': network.module.embedding.parameters()},        
                              ], lr=args.lr)


if not os.path.exists(args.save_dir+"/test"):
    os.makedirs(args.save_dir+"/test")

wandb.init(project="StyTR",name="StyTR")

content_loss_sum = 0.0
style_loss_sum = 0.0
total_image_loss_sum = 0.0
feature_loss_sum = 0.0
total_loss_sum = 0.0
for i in tqdm(range(args.max_iter)):

    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)  
    out, loss_c, loss_s, total_image_loss, feature_loss = network(content_images, style_images)

    if i % 100 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
                        args.save_dir, str(i),".jpg"
                    )
        out = torch.cat((content_images,out),0)
        out = torch.cat((style_images,out),0)
        save_image(out, output_name)

        
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + (total_image_loss * 70) + (feature_loss * 1) 
  
    print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
              ,"-l1:",total_image_loss.sum().cpu().detach().numpy(),"-l2:",feature_loss.sum().cpu().detach().numpy()
              )
    content_loss_sum += loss_c.sum().cpu().detach().numpy()
    style_loss_sum += loss_s.sum().cpu().detach().numpy()
    total_image_loss_sum += total_image_loss.sum().cpu().detach().numpy()
    feature_loss_sum += feature_loss.sum().cpu().detach().numpy()
    total_loss_sum += loss.sum().cpu().detach().numpy()

    if i % 50 == 0 and i != 0:
        wandb.log({"content_loss":content_loss_sum/50,"style_loss":style_loss_sum/50,"total_image_loss":total_image_loss_sum/50,"feature_loss":feature_loss_sum/50,"total_loss":total_loss_sum/50, 'activation_func': args.activation_func, 'batch_size': args.batch_size, 'hidden_dimension': args.hidden_dim, 'lr': args.lr})
        content_loss_sum = 0.0
        style_loss_sum = 0.0
        total_image_loss_sum = 0.0
        feature_loss_sum = 0.0
        total_loss_sum = 0.0
       
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', total_image_loss.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', feature_loss.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.module.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = network.module.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.module.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

                                                    
writer.close()
wandb.finish()


