import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import re
import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def get_gpu_memory():
    if torch.cuda.is_available():
        return f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    return "GPU not available"

# Print initial VRAM usage
print("Initial", get_gpu_memory())




# other models

import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(DEVICE)

processor = AutoProcessor.from_pretrained(model_id)




















# if need to download
# Initialize model and processor globally
# processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
# model = AutoModelForVision2Seq.from_pretrained(
#     "HuggingFaceTB/SmolVLM-Instruct",
#     torch_dtype=torch.bfloat16,
#     _attn_implementation="eager"
# ).to(DEVICE)
    











# if already downloaded
# After model loading: GPU memory allocated: 4.19 GB
# During inference: GPU memory allocated: 4.24 GB


# MODEL_PATH = "/data/students/christian/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-Instruct/snapshots/7fb3550ea09521c12c2d17026b537f86e083e8aa"
# MODEL_PATH = "/home/dominic/Desktop/me7/SmolVLM"
# MODEL_PATH = "/home/dominic/Desktop/me7/Paligemma" # causes crashing of jetson

# processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
# model = AutoModelForVision2Seq.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.bfloat16,
#     _attn_implementation="eager",
#     local_files_only=True
# ).to(DEVICE)










# if quantization is needed
# After model loading: GPU memory allocated: 1.45 GB
# During inference: GPU memory allocated: 1.50 GB



# # Configure 4-bit quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )


# # if need to download
# # Initialize model and processor globally
# processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
# model = AutoModelForVision2Seq.from_pretrained(
#     "HuggingFaceTB/SmolVLM-Instruct",
#     quantization_config=quantization_config,
#     _attn_implementation="eager"
# ).to(DEVICE)
    
    
    
# # if already downloaded
# # MODEL_PATH = "/data/students/christian/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-Instruct/snapshots/7fb3550ea09521c12c2d17026b537f86e083e8aa"
# MODEL_PATH = "/home/dominic/Desktop/me7/SmolVLM"
# processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
# model = AutoModelForVision2Seq.from_pretrained(
#     MODEL_PATH,
#     quantization_config=quantization_config,
#     _attn_implementation="eager",
#     local_files_only=True
# ).to(DEVICE)







# Print VRAM usage after model loading
print("After model loading:", get_gpu_memory())



# to print model size
# cmd: du -sh /data/students/christian/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-Instruct


# Global variable to store the latest webcam frame as a NumPy array
latest_frame_np = None

def update_latest_frame_from_stream(frame_np):
    global latest_frame_np
    latest_frame_np = frame_np
    return frame_np  # Return the frame to keep the stream going

def process_image(prompt_text):
    global latest_frame_np
    if latest_frame_np is None:
        return "No image captured yet from the webcam."

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(latest_frame_np)

    # Create input messages
    messages = [
        {
            "role": "system",
            "content": """You are a versatile visual AI assistant capable of analyzing images and answering any questions about them.
            You can describe scenes, analyze objects, interpret actions, answer questions about visual elements, make comparisons,
            and provide insights based on what you see. Be precise and confident in your knowledge while being honest about any uncertainties.
            Focus on providing relevant information that directly addresses the user's question.
            
            Format your responses in clear, well-structured paragraphs for better readability. 
            
            Don't put too many sentences in a single paragraph."""
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[pil_image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # generated_text = re.sub(r"^.*?Assistant:", "", generated_text, flags=re.DOTALL).strip()
    generated_text = re.sub(r"^.*?[Aa]ssistant ?: ?", "", generated_text).strip()

    print("During inference:", get_gpu_memory())

    return generated_text

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        # Use Image component with streaming=True and type="numpy"
        webcam = gr.Image(sources=["webcam"], type="numpy", streaming=True)

        with gr.Column():
            text_input = gr.Textbox(
                label="Question",
                placeholder="Ask something about the image...",
                value="Can you describe this image?"
            )
            text_output = gr.Textbox(label="Response")

    btn = gr.Button("Submit Query")
    btn.click(
        fn=process_image,
        inputs=[text_input],
        outputs=text_output
    )

    # Continuously update the latest frame from the webcam stream
    webcam.stream(update_latest_frame_from_stream, inputs=webcam, outputs=webcam,
                stream_every=0.6, # BEST SO FAR IN CONDO
                                    # # use this for now, in order to still have a decent FPS
                                    # 1.6 fps
                # stream_every=0.075, # BEST SO FAR IN EDUROAM, 0.033 still laggy
                )

# Launch the interface
demo.launch()
