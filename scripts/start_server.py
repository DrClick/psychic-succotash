# This is needed to keep the code functioning with imports correctly and settings
import os
import sys

# Add the root directory (parent of scripts) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------

from fastapi import FastAPI, File, Form, UploadFile, WebSocket
import asyncio
import logging
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import tempfile
import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import mode
import tensorflow as tf
import uuid

from classical_composer.data import generate_piano_roll
from classical_composer.utility import create_piano_roll_image
from classical_composer.frames import (
    extract_frames,
    generate_random_frame_indicies,
    get_fixed_frame_indicies,
)
from classical_composer.features.feature_extractor import FeatureExtractor
from settings import settings
from classical_composer.misc.logging import logger


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
logger.info(f"Mounting static files from: {settings.static_folder}")
app.mount("/static", StaticFiles(directory=settings.static_folder), name="static")


# load in the models
model_location = f"{settings.resource_folder}/{settings.kmeans_model}"
logger.info(f"loading KMeans model from: {model_location}")
with open(f"{settings.resource_folder}/{settings.kmeans_model}", "rb") as f:
    kmeans_model_run = pickle.load(f)
    featureExtractor = FeatureExtractor()

model_location = f"{settings.resource_folder}/{settings.cnn_model}"
logger.info(f"loading CNN model from: {model_location}")
with open(model_location, "rb") as f:
    with tf.device('/CPU:0'):
        cnn_model = tf.keras.models.load_model(f"{settings.resource_folder}/{settings.cnn_model}")



# Route to process MIDI file
@app.post("/predict/")
async def predict(file: UploadFile = File(...), task_id: str = Form(...)):
    task_id = str(uuid.uuid4())  # Unique identifier for this request
    try:
        logger.info(f"Task {task_id}: Processing MIDI file")
        result, input_piano_roll, extracted_frames = await asyncio.to_thread(
            process_midi_file, file
        )

        # Add predictions to the result
        logger.info(f"Task {task_id}: Predicting KMeans")
        kmeans_prediction = await asyncio.to_thread(
            predict_kmeans, extracted_frames, kmeans_model_run, featureExtractor
        )
        result["KMEANS_prediction"] = kmeans_prediction


        logger.info(f"Task {task_id}: Predicting CNN")
        cnn_prediction = await asyncio.to_thread(
            predict_cnn, extracted_frames, cnn_model
        )
        result["CNN_prediction"] = cnn_prediction

        # Include task ID in the response for WebSocket logging
        result["task_id"] = task_id
        return result

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


def process_midi_file(file):
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

    # fix up filepaths to URLs
    result["file_name"] = f"{settings.base_url}/static/{result['file_name']}"
    for x in result["frames"]:
        x["file_name"] = f"{settings.base_url}/static/{x['file_name']}"

    # Cleanup temporary file
    os.unlink(temp_file.name)
    return result, input_piano_roll, extracted_frames


def predict_kmeans(extracted_frames, kmeans_model_run, featureExtractor):
    # Predict using KMeans
    kmeans = kmeans_model_run["model"]
    scaler = kmeans_model_run["scaler"]
    cluster_mapping = kmeans_model_run["cluster_mapping"]
    threshold = kmeans_model_run["threshold"]
    feature_columns = featureExtractor.feature_names

    predictions = []
    for idx, frame in enumerate(extracted_frames):
        features = featureExtractor.extract_all_features(frame)
        features["frame_id"] = idx
        features_df = pd.DataFrame(features, index=["frame_id"])[feature_columns]

        #normalize features
        X = scaler.transform(features_df)    
        distances = kmeans.transform(X)  # Distance to each centroid
        closest_clusters = np.argmin(distances, axis=1)  # Index of closest cluster
        
        # Map clusters to composers
        prediction = [cluster for cluster in closest_clusters][0]
        if distances[0,closest_clusters] > threshold:
            prediction = -1
        else:
            prediction = closest_clusters[0]
        predictions.append(prediction)

    final_result = mode(predictions)
    final_class = cluster_mapping.get(final_result, "unknown")
    return final_class


def predict_cnn(extracted_frames, cnn_model):
    with tf.device('/CPU:0'):
        X = np.expand_dims(extracted_frames, axis=-1)
        scores = cnn_model.predict(X)
        threshold = 0.5  # minimum threshold for classification
        predictions = []
        for score in scores:
            max_value = np.max(score)
            if max_value > threshold:
                predictions.append(int(np.argmax(score)))
            else:
                predictions.append(-1)
    final_result = int(mode(predictions)[0])
    final_class = ['Bach', 'Beethoven', 'Brahms', 'Schubert', 'Unknown'][final_result]
    return final_class

# WebSocket endpoint
# Global list of connected WebSockets
connected_clients = []

@app.websocket("/logs/{task_id}")
async def websocket_logs(websocket: WebSocket, task_id: str):
    await websocket.accept()

    # Create a task-specific logger for this WebSocket
    log_handler = TaskWebSocketLogHandler(websocket)
    log_handler.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    try:
        # Keep the WebSocket connection alive during processing
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket {task_id} disconnected: {e}")
    finally:
        # Remove the task-specific logger when done
        logger.removeHandler(log_handler)
        await websocket.close()

class TaskWebSocketLogHandler(logging.Handler):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket

    def emit(self, record):
        log_entry = self.format(record)
        asyncio.ensure_future(self.send_log(log_entry))

    async def send_log(self, log_entry):
        try:
            await self.websocket.send_text(log_entry)
        except Exception as e:
            pass

if __name__ == "__main__":
    """Run the FastAPI server with Uvicorn web server."""
    import uvicorn


    uvicorn.run(app, host="0.0.0.0", port=5000)
