from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import model as model_logic # This imports your new model.py
import uvicorn # For running the server

# Initialize the FastAPI app
app = FastAPI(title="LMS Caption Generator API")

# --- Load the Model on Startup ---
# This loads the BLIP model once when the server starts
try:
    model, processor = model_logic.load_model()
    print("AI Model loaded and ready.")
except Exception as e:
    print(f"FATAL: Model failed to load: {e}")
    model = None
    processor = None

# --- Configure CORS (Cross-Origin Resource Sharing) ---
# This allows your React app (localhost:3000) to talk to this server (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the BLIP Image Captioning API"}

@app.post("/generate-caption")
async def generate_caption_endpoint(file: UploadFile = File(...)):
    """
    Receives an uploaded image, generates a detailed caption, and returns it.
    """
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load on startup.")

    # 1. Read the image file from the request
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    # 2. Generate the caption using our model
    try:
        # This function call is now simpler
        caption = model_logic.predict_caption(image) 
        
        # 3. Return the caption
        return {"caption": caption}
    
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate caption.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)