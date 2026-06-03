import torch
import clip
from PIL import Image

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def embed_image(image_path_or_pil):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0]