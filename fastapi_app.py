import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from accelerate import Accelerator
from net import (
    SUM,
    load_and_preprocess_image,
    predict_saliency_map,
    overlay_heatmap_on_image,
    write_heatmap_to_image,
)
from inference import setup_model
from fastapi import Security
from fastapi.security.api_key import APIKeyHeader
from fastapi import Depends
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SUM_API_KEY")

API_KEY_NAME = "API-KEY"
api_key_header = APIKeyHeader(
    name=API_KEY_NAME, scheme_name="Standard API scheme", auto_error=False
)


def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key


app = FastAPI(
    title="SUM Saliency Map Prediction",
    description="Upload an image to generate its saliency heatmap.",
)

# Setup the accelerator and model once at startup.
accelerator = Accelerator()
model = setup_model(accelerator.device)
# Alternatively, you might use:
# model = SUM.from_pretrained("safe-models/SUM").to(accelerator.device)


def predict(image_path: str, condition: int):
    """
    Process the image to produce both an overlay and a saliency heatmap.
    Returns a tuple: (overlay_image_path, heatmap_image_path).
    """
    # Create filenames based on the input image name.
    filename = os.path.splitext(os.path.basename(image_path))[0]
    hot_output_filename = f"{filename}_saliencymap.png"
    overlay_output_filename = f"{filename}_overlay.png"

    # Preprocess the image, predict saliency, and create the heatmap and overlay.
    image, orig_size = load_and_preprocess_image(image_path)
    saliency_map = predict_saliency_map(image, condition, model, accelerator.device)
    write_heatmap_to_image(saliency_map, orig_size, hot_output_filename)
    overlay_heatmap_on_image(image_path, hot_output_filename, overlay_output_filename)

    return overlay_output_filename, hot_output_filename


@app.post("/predict", summary="Generate saliency heatmap")
async def predict_endpoint(
    file: UploadFile = File(..., description="Input image file"),
    condition: int = Form(2, description="Condition integer (e.g., 0, 1, 2, or 3)"),
    api_key: str = Depends(get_api_key),
):
    """
    Accepts an image and a condition, and returns the saliency heatmap (as a PNG image).
    """
    # Save the uploaded file to a temporary location.
    try:
        temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, temp_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {e}"
        )

    # Run the prediction pipeline.
    try:
        overlay_file, heatmap_file = predict(file_path, condition)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Return the heatmap image as the response.
    # (Alternatively, you can return overlay_file or even both as part of a JSON response.)
    if not os.path.exists(heatmap_file):
        raise HTTPException(status_code=500, detail="Heatmap file not found.")

    return FileResponse(
        heatmap_file, media_type="image/png", filename=os.path.basename(heatmap_file)
    )


# To run the app using: uvicorn my_fastapi_app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, workers=15)  # 12gb vram
