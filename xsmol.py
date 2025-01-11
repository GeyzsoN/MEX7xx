import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import os

# Force GPU #0 to be visible (and used if there's enough memory)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load images
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")

# Initialize processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

# Initialize model, forcing all weights on GPU index 0
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    device_map={"": 0},  # all layers on GPU 0
    # Remove or replace _attn_implementation
)

print(f"Model has {model.num_parameters():,} parameters")

# ---- Remove manual model.to(DEVICE) here ----
# model.to(DEVICE)  # <- This was causing offload conflicts with accelerate
print("Model loaded to device (fully on GPU)")

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {
                "type": "text",
                "text": "Can you describe the two images in not so great detail? "
                        "make it sound like you are teasing what's in the image"
            }
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500, synced_gpus=False)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
