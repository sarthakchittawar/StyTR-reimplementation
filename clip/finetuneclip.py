import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.load_state_dict(torch.load("model_4.pt"))

print("Model Loaded")

class image_caption_dataset(Dataset):
    def __init__(self, image_paths, image_captions, preprocess):
        self.image_paths = image_paths
        self.image_captions = clip.tokenize(image_captions)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_paths[idx]))
        caption = self.image_captions[idx]
        return image, caption
    

with open("descriptions.json") as f:
    descriptions = json.load(f)

image_captions = []
image_paths = []
for description in descriptions:
    path = f"/scratch/sanika/wikiart/{description['filename']}"
    image_captions.append(description["description"])
    image_paths.append(path)

dataset = image_caption_dataset(image_paths, image_captions, preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Dataloader created")

def convert_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

optimizer = torch.optim.Adam(model.parameters(), lr=4e-5,betas=(0.9,0.98),eps=1e-5)
loss_image = nn.CrossEntropyLoss()
loss_caption = nn.CrossEntropyLoss()

epochs = 5

for epoch in range(epochs):
    loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        images, captions = batch
        images = images.to(device)
        captions = captions.to(device)

        logits_per_image, logits_per_text = model(images, captions)

        ground = torch.arange(len(images), dtype=torch.long).to(device)
        total_loss = loss_image(logits_per_image, ground) + loss_caption(logits_per_text, ground)
        total_loss.backward()
        convert_to_fp32(model)
        optimizer.step() 
        clip.model.convert_weights(model)
        loss += total_loss.item()

    # save model as epoch
    torch.save(model.state_dict(), f"model_{epoch + 5}.pt")

    print("Epoch: {} Loss: {}".format(epoch, loss))



# loss went down from like 7000? to 1000 in 10 epochs