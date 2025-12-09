import os
import sys
import json
import mlflow
import mlflow.keras
import tensorflow as tf
import numpy as np
import h5py
from datetime import datetime

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Prefer the standard MLflow env var used in your docker-compose
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_URI", "http://localhost:5000"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/crnn_net_gru_adam_ours_epoch_40.h5")
MODEL_NAME = os.getenv("MODEL_NAME", "MelodyMind_CRNN")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Audio_Classification")

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------

def build_model(num_genres: int = len(GENRES)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, 1440, 1), name="input_1")
    
    x = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)), name="zeropadding2d_1")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=2, name="bn_0_freq")(x)

    # Conv Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ELU(name="elu_1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="pool1")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout1")(x)

    # Conv Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ELU(name="elu_2")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), padding="same", name="pool2")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout2")(x)

    # Conv Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ELU(name="elu_3")(x)
    x = tf.keras.layers.MaxPooling2D((4, 2), padding="same", name="pool3")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout3")(x)

    # Conv Block 4
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.ELU(name="elu_4")(x)
    x = tf.keras.layers.MaxPooling2D((4, 2), padding="same", name="pool4")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout4")(x)

    # Reshape
    x = tf.keras.layers.Permute((1, 3, 2), name="permute_1")(x)
    x = tf.keras.layers.Reshape((-1, 128), name="reshape_1")(x)

    # RNN Layers
    x = tf.keras.layers.GRU(32, return_sequences=True, name="gru1")(x)
    x = tf.keras.layers.GRU(32, return_sequences=False, name="gru2")(x)
    
    # Output
    x = tf.keras.layers.Dropout(0.3, name="final_drop")(x)
    outputs = tf.keras.layers.Dense(num_genres, activation="softmax", name="preds")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="genre_classifier")

# ------------------------------------------------------------------
# Manual weight loading (unchanged logic)
# ------------------------------------------------------------------

def manual_weight_loading(model, filepath):
    print(f"Opening H5 file: {filepath}")
    with h5py.File(filepath, 'r') as f:
        for layer in model.layers:
            if not layer.weights:
                continue
            
            layer_name = layer.name
            if layer_name not in f.attrs['layer_names'].astype(str):
                continue

            print(f"Loading weights for layer: {layer_name}...")
            g = f[layer_name]
            weights = [g[w_name] for w_name in g.attrs['weight_names']]
            weight_values = [w[:] for w in weights]
            
            # Conv2D kernel transpose if needed
            if isinstance(layer, tf.keras.layers.Conv2D):
                kernel = weight_values[0]
                if kernel.shape != layer.weights[0].shape:
                    print(f"  -> Transposing Conv2D kernel from {kernel.shape}")
                    weight_values[0] = np.transpose(kernel, (2, 3, 1, 0))

            # GRU special handling (9 weight tensors -> 3 Keras tensors)
            if isinstance(layer, tf.keras.layers.GRU) and len(weight_values) == 9:
                print(f"  -> Merging 9 interleaved GRU weights...")
                
                kernel = np.concatenate(
                    [weight_values[0], weight_values[3], weight_values[6]], axis=-1
                )
                recurrent_kernel = np.concatenate(
                    [weight_values[1], weight_values[4], weight_values[7]], axis=-1
                )
                bias = np.concatenate(
                    [weight_values[2], weight_values[5], weight_values[8]], axis=-1
                )
                
                target_bias_shape = layer.weights[2].shape
                if len(target_bias_shape) == 2:
                    print(f"  -> Adapting bias to shape {target_bias_shape}")
                    new_bias = np.zeros(target_bias_shape)
                    new_bias[0] = bias
                    bias = new_bias
                
                weight_values = [kernel, recurrent_kernel, bias]

            layer.set_weights(weight_values)

# ------------------------------------------------------------------
# MLflow helper: log params & artifacts around registration
# ------------------------------------------------------------------

def log_model_metadata_and_artifacts(model):
    """
    Log params and artifacts related to the registered model.
    """
    # --------- Parameters ---------
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "num_genres": len(GENRES),
        "genres": ",".join(GENRES),
        "framework": "tensorflow.keras",
        "architecture": "CRNN_GRU",
    })

    # --------- Artifacts directory ---------
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1) Model summary
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    summary_text = "\n".join(summary_lines)

    summary_path = os.path.join(artifacts_dir, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    # 2) Label / genre mapping
    label_mapping_path = os.path.join(artifacts_dir, "label_mapping.json")
    with open(label_mapping_path, "w") as f:
        json.dump({"genres": GENRES}, f, indent=2)

    # 3) Run info / metadata
    run_info_path = os.path.join(artifacts_dir, "run_info.txt")
    with open(run_info_path, "w") as f:
        f.write(f"Model Name: {MODEL_NAME}\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Registered At: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Num Genres: {len(GENRES)}\n")
        f.write(f"Genres: {', '.join(GENRES)}\n")

    # Log the entire artifacts folder
    mlflow.log_artifacts(artifacts_dir)

# ------------------------------------------------------------------
# Main registration function
# ------------------------------------------------------------------

def upload_and_register():
    print(f"Connecting to MLflow at {MLFLOW_URI}...")
    mlflow.set_tracking_uri(MLFLOW_URI)

    print(f"1. Building CRNN Architecture...")
    model = build_model()
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"2. Loading & Fixing weights from {MODEL_PATH}...")
    try:
        manual_weight_loading(model, MODEL_PATH)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Critical Weight Loading Error: {e}")
        sys.exit(1)

    print("3. Setting MLflow experiment...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("4. Starting MLflow run, logging params, artifacts and registering model...")
    with mlflow.start_run() as run:
        # (a) log params + artifacts
        log_model_metadata_and_artifacts(model)

        # (b) log and register the Keras model
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print(f"Success! Model Registered. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    upload_and_register()
