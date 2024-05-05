import json
import torch
import torch.nn as nn
from PIL import Image
import clip
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import stytr as StyTR
import transformer as transformer
from os.path import basename
from os.path import splitext
from torchvision.utils import save_image


# check if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the clip model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.load_state_dict(torch.load("clip/model_9.pt"))

# load the clip encodings
encodings = torch.load("clip/encodings.pt")
encodings = torch.stack(encodings)
encodings = encodings.squeeze(1)

# loading descriptions
with open("clip/descriptions.json") as f:
    descriptions = json.load(f)

image_captions = []
image_paths = []
for description in descriptions:
    path = f"/scratch/sanika/wikiart/{description['filename']}"
    image_captions.append(description["description"])
    image_paths.append(path)

def get_top_image(text):
    text_features = model.encode_text(clip.tokenize([text]).to(device))
    similarities = F.cosine_similarity(encodings, text_features, dim=-1)
    # Get top 5 indices
    values, indices = similarities.topk(1)
    print("Top image retrieved.")
    # Get top 5 image paths
    return [image_paths[i] for i in indices][0]

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

output_path = "output"

def style_transfer(style_path, content_path, user_input):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("Beginning style transfer...")
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load('experiments/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder_arch
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load('/scratch/sanika/experiments/decoder_iter_160000.pth')
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load('/scratch/sanika/experiments/transformer_iter_160000.pth')
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load('/scratch/sanika/experiments/embedding_iter_160000.pth')
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg,decoder,embedding,Trans, None)
    network.eval()
    network.to(device)



    content_tf = test_transform(512, 'store_true')
    style_tf = test_transform(512, 'store_true')
 

    # content_tf = content_transform()       
    content = content_tf(Image.open(content_path).convert("RGB"))

    h,w,c=np.shape(content)    
    # style_tf = style_transform(h,w)
    style = style_tf(Image.open(style_path).convert("RGB"))

    
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    
    with torch.no_grad():
        output= network(content,style)       
    output = output[0].cpu()
    
    with torch.no_grad():
        output= network(content,style)       
    output = output[0].cpu()

    # add the description (_ instead of space to the output name) 
    output_name = f"{output_path}/{splitext(basename(content_path))[0]}_{user_input.replace(' ', '_')}.png"
    # add the original image name as well
    save_image(output, output_name)



if __name__ == "__main__":
    user_input = input("Enter a description of the style you want to apply: ")
    content_path = input("Enter the path of the content image: ")
    image_path = get_top_image(user_input)
    style_transfer(image_path, content_path, user_input)
    print("Saved at output/")