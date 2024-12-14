import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

def capture_image_from_webcam():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return None

    print("Webcam is ready. Press Enter to capture an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        # Show the webcam feed
        cv2.imshow("Webcam Feed", frame)

        # Wait for Enter key to capture the image
        if cv2.waitKey(1) & 0xFF == 13:  # ASCII for Enter key
            cap.release()
            cv2.destroyAllWindows()
            # Convert BGR (OpenCV format) to RGB (PIL format)
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Release webcam and close window if no image captured
    cap.release()
    cv2.destroyAllWindows()
    return None

# Capture a single image from webcam
print("Opening webcam to capture an image...")
webcam_image = capture_image_from_webcam()

if webcam_image is not None:
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe the image?"}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[webcam_image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("Model Output:")
    print(generated_texts[0])
else:
    print("Image was not captured. Please try again.")