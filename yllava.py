import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Log model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Check CUDA memory usage after model loading
print(f"Memory allocated (MB): {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f}")
print(f"Max memory allocated (MB): {torch.cuda.max_memory_allocated(0) / (1024 ** 2):.2f}")

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Load image
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

# Check CUDA memory usage after inputs are prepared
print(f"Memory allocated after inputs (MB): {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f}")
print(f"Max memory allocated after inputs (MB): {torch.cuda.max_memory_allocated(0) / (1024 ** 2):.2f}")

# Generate output
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

# Check CUDA memory usage after inference
print(f"Memory allocated after inference (MB): {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f}")
print(f"Max memory allocated after inference (MB): {torch.cuda.max_memory_allocated(0) / (1024 ** 2):.2f}")

# Decode and print result
print(processor.decode(output[0][2:], skip_special_tokens=True))
