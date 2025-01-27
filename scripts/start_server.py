# This is needed to keep the code functioning with imports correctly and settings
import os
import sys


# Add the root directory (parent of scripts) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import tempfile
import os

# Import your trained models and utilities
from classical_composer.data import generate_piano_roll
from classical_composer.utility import create_piano_roll_image
from classical_composer.frames import (
    extract_frames,
    generate_random_frame_indicies,
    get_fixed_frame_indicies,
)

from settings import settings


app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with the frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Serve static files (Generated piano roll images)
app.mount("/static", StaticFiles(directory=settings.static_folder), name="static")


# Route to process MIDI file
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary directory
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        with temp_file as f:
            f.write(file.file.read())

        # Generate piano roll
        input_piano_roll = generate_piano_roll(temp_file.name)

        # extract random frames
        fs = 100
        frame_size = 3000
        fixed_frames = get_fixed_frame_indicies(input_piano_roll.shape, frame_size=frame_size)

        # Extract random frames from the piano roll
        random_frames = generate_random_frame_indicies(
            input_piano_roll.shape,
            n_frames=20,
            frame_size=frame_size,
            buffer_size=5,
            random_seed=59,
        )
        frame_indicies = [(int(start), int(end)) for start, end in fixed_frames + random_frames]
        extracted_frames = extract_frames(input_piano_roll, frame_indicies)
        # # Plot quality check
        result = create_piano_roll_image(
            input_piano_roll, extracted_frames, frame_indicies, settings.static_folder, fs=100
        )

        result["CNN_prediction"] = "UNKNOWN"
        result["KMEANS_prediction"] = "UNKNOWN"

        # fix up filepaths to URLs
        result["file_name"] = f"{settings.base_url}/static/{result['file_name']}"
        for x in result["frames"]:
            x["file_name"] = f"{settings.base_url}/static/{x['file_name']}"

        # Cleanup temporary file
        os.unlink(temp_file.name)

        # Return predictions and image path
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
