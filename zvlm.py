import torch
import open_clip
from PIL import Image
import requests

# Load a lightweight CLIP model (ResNet-50 backbone, which has <1B parameters)
model_name = "RN50"  # ResNet-50 backbone (efficient and small)
pretrained = "openai"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load the model and tokenizer
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters")
model = model.to(device)
print("Model loaded to device")

# Test image and text inputs
image_url = "https://images.unsplash.com/photo-1581093588401-8343e4aa6a36"  # Sample image URL
text_inputs = ["a dog", "a cat", "a person"]  # Example text prompts

print("Downloading and preprocessing the image...")
# Preprocess the image
image = Image.open(requests.get(image_url, stream=True).raw)
image_tensor = preprocess(image).unsqueeze(0).to(device)

print("Downloading and tokenizing the text...")
# Tokenize the text
text_tokens = tokenizer(text_inputs).to(device)

print("Performing inference...")
# Perform inference
with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tokens)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = image_features @ text_features.T
    print("Similarity Scores:", similarity.squeeze().cpu().numpy())
