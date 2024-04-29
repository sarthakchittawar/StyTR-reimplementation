import json
import torch
import torch.nn as nn
from PIL import Image
import clip
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.load_state_dict(torch.load("model_9.pt"))


encodings = torch.load("encodings.pt")
encodings = torch.stack(encodings)
encodings = encodings.squeeze(1)

with open("descriptions.json") as f:
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
    print(similarities.shape)
    # Get top 5 indices
    values, indices = similarities.topk(1)

    # Get top 5 image paths
    return [image_paths[i] for i in indices][0]

print(get_top_image("colorful and happy")) 