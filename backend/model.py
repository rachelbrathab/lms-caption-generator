from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Global variables to hold the loaded model
model = None
processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """
    Loads the BLIP model and processor onto the available device.
    This is called once when the FastAPI server starts.
    """
    global model, processor
    
    print(f"Loading BLIP model onto {device}...")
    
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    
    print("BLIP Model loaded successfully.")
    return model, processor

def predict_caption(image: Image, gen_kwargs=None):
    """
    Generates a detailed caption for the given image.
    This is the function our API will call.
    """
    if not model or not processor:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
        
    if gen_kwargs is None:
        # These settings produce more detailed captions
        gen_kwargs = {"max_length": 50, "num_beams": 5, "no_repeat_ngram_size": 2}

    # Pre-process the image
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
        
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate the caption (token IDs)
    output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode the token IDs to text
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    
    # Clean up the output
    caption = caption.strip()
    
    return caption

if __name__ == "__main__":
    # A simple test to run this file directly
    # python model.py
    
    load_model()
    # Test with a placeholder image (requires PIL to create)
    try:
        test_image = Image.new('RGB', (200, 200), color = 'red')
        print("--- Running test prediction ---")
        test_caption = predict_caption(test_image)
        print(f"Test Image (red box) caption: {test_caption}")
    except Exception as e:
        print(f"Test failed: {e}")