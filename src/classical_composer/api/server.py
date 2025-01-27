from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
import tempfile
import os

# Import your trained models and utilities
from classical_composer.data import generate_piano_roll
from PIL import Image

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()

# Serve static files (Generated piano roll images)
app.mount("/static", StaticFiles(directory=f"{BASE_DIR}/static"), name="static")


# Route to process MIDI file
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary directory
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        with temp_file as f:
            f.write(file.file.read())

        # Generate piano roll
        piano_roll = generate_piano_roll(temp_file.name)

        # Save the piano roll image
        output_image_path = Path("static") / f"{Path(temp_file.name).stem}_piano_roll.png"
        plt.figure(figsize=(10, 6))
        plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="hot")
        plt.title("Piano Roll")
        plt.ylabel("Pitch")
        plt.xlabel("Time (frames)")
        plt.colorbar(label="Velocity")
        plt.savefig(output_image_path)
        plt.close()

        # Process features for K-Means and CNN
        piano_roll_flat = piano_roll.sum(axis=0).reshape(1, -1)  # Example feature transformation
        kmeans_input = scaler.transform(piano_roll_flat)
        kmeans_prediction = kmeans_model.predict(kmeans_input)[0]
        kmeans_composer = composer_labels[kmeans_prediction]

        cnn_prediction = cnn_model.predict(piano_roll[np.newaxis, :, :, np.newaxis])
        cnn_composer = composer_labels[np.argmax(cnn_prediction)]

        # Cleanup temporary file
        os.unlink(temp_file.name)

        # Return predictions and image path
        return {
            "piano_roll_image": f"/static/{output_image_path.name}",
            "kmeans_composer": kmeans_composer,
            "cnn_composer": cnn_composer,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
