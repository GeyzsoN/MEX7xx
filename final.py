import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import cv2
import os

# Set CUDA environment and device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load model and processor
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

# Log model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Check CUDA memory usage after model loading
print(f"Memory allocated (MB): {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f}")
print(f"Max memory allocated (MB): {torch.cuda.max_memory_allocated(0) / (1024 ** 2):.2f}")

# Function to capture an image from the webcam
def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return None
    
    print("Press 's' to capture an image and 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        cv2.imshow("Webcam Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the image on 's' key press
            cap.release()
            cv2.destroyAllWindows()
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif key == ord('q'):  # Quit on 'q' key press
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

# Capture an image from the webcam
print("Opening webcam to capture images...")
webcam_image = capture_image_from_webcam()

if webcam_image is not None:
    print("Image captured successfully.")
    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare inputs
    print("Processing the captured image...")
    inputs = processor(images=webcam_image, text=prompt, return_tensors="pt").to(0, torch.float16)

    # Check CUDA memory usage after inputs are prepared
    print(f"Memory allocated after inputs (MB): {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f}")
    print(f"Max memory allocated after inputs (MB): {torch.cuda.max_memory_allocated(0) / (1024 ** 2):.2f}")

    # Generate output
    print("Generating outputs...")
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    # Check CUDA memory usage after inference
    print(f"Memory allocated after inference (MB): {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f}")
    print(f"Max memory allocated after inference (MB): {torch.cuda.max_memory_allocated(0) / (1024 ** 2):.2f}")

    # Decode and print result
    print("Decoding output...")
    print(processor.decode(output[0][2:], skip_special_tokens=True))
else:
    print("No image was captured. Exiting.")
