import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
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
    
    print("Press 's' to capture an image and 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        # Show the webcam feed
        cv2.imshow("Webcam Feed", frame)

        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the image on 's' key press
            cap.release()
            cv2.destroyAllWindows()
            # Convert BGR (OpenCV format) to RGB (PIL format)
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif key == ord('q'):  # Quit on 'q' key press
            break

    # Release webcam and close window
    cap.release()
    cv2.destroyAllWindows()
    return None

# Capture images from webcam
print("Opening webcam to capture images...")
webcam_image1 = capture_image_from_webcam()
# webcam_image2 = capture_image_from_webcam()

print("Processing images...")
if webcam_image1 is not None:
    # Create input messages
    print("Images captured successfully.")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                # {"type": "image"},
                {"type": "text", "text": "Can you describe the two images?"}
            ]
        },
    ]

    print("Generating description...")
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[webcam_image1], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    print("Generating description...")
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("Model Output:")
    print(generated_texts[0])
else:
    print("Images were not captured. Please try again.")