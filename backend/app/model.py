import os
import tensorflow as tf
import numpy as np
import librosa
import mlflow # NEW: Import MLflow to talk to the model server

# Get directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NEW: MLOps Configuration
# We get the MLflow server URL from the environment variable (set in Docker Compose)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "MelodyMind_CRNN"
STAGE = "Production"

# List of genres your model predicts
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# Internal global model instance (lazy-loaded)
_model: tf.keras.Model | None = None


def build_model(num_genres: int = len(GENRES)) -> tf.keras.Model:
    """Build a simple CRNN-style model compatible with the features
    produced by extract_features.

    Input shape: (time_steps, 128) where 128 is the number of mel bins.
    """
    # time_steps is variable (None) because clips can have different lengths
    inputs = tf.keras.Input(shape=(None, 128), name="mel_spectrogram")

    # Mask padding timesteps (if any)
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)

    # Recurrent layers over time dimension
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, return_sequences=True)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64)
    )(x)

    # Dense layers for classification
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_genres, activation="softmax", name="genre_logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="genre_classifier")
    return model


def get_model() -> tf.keras.Model:
    """Return a singleton model instance.

    NEW: Tries to load the 'Production' model from the MLflow Model Registry.
    If the MLflow server is down or the model isn't found, it falls back
    to a randomly initialised model.
    """
    global _model

    if _model is None:
        # NEW: MLOps Logic Starts Here
        print(f"[MLOps] Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Construct the URI: models:/<ModelName>/<Stage>
        model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        
        print(f"[MLOps] Attempting to load '{MODEL_NAME}' from stage '{STAGE}'...")
        
        try:
            # NEW: Download and load the model from the registry
            _model = mlflow.keras.load_model(model_uri)
            print("[MLOps] Success! Production model loaded from MLflow.")
            
        except Exception as e:
            print(f"[MLOps] Error loading from MLflow: {e}")
            print("[MLOps] CRITICAL: Fallback to random weights.")
            # Fallback to the architecture defined in code (random weights)
            _model = build_model()

    return _model


def extract_features(file_path: str,
                     sr: int = 22050,
                     n_mels: int = 128) -> np.ndarray:
    """Load an audio file and convert it to a log-mel spectrogram.

    Returns a 3D numpy array of shape (1, time_steps, n_mels), suitable as
    input to the model returned by get_model().
    """
    # Load mono audio at the target sampling rate
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    # Convert power spectrogram (amplitude squared) to decibel (log) units
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Shape now: (n_mels, time_steps) -> transpose to (time_steps, n_mels)
    log_mel_spec = log_mel_spec.T.astype("float32")

    # Add batch dimension and channel dim -> (1, time_steps, n_mels, 1)
    # This matches Conv2D-based models that expect a final channel dimension.
    return np.expand_dims(log_mel_spec, axis=(0, -1))


def predict_genre(file_path: str) -> str:
    """Predict the genre of an audio file.

    This function is what your FastAPI endpoint should call.
    """
    model = get_model()
    features = extract_features(file_path)

    # Run inference
    probs = model.predict(features)[0]  # shape: (num_genres,)
    predicted_index = int(np.argmax(probs))

    # Safety check in case of any mismatch
    if predicted_index < 0 or predicted_index >= len(GENRES):
        raise ValueError(f"Predicted index {predicted_index} is out of bounds for GENRES list.")

    return GENRES[predicted_index]