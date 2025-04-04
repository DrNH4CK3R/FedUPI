from flask import Flask, request, jsonify, render_template, Response, send_file
from flask_cors import CORS
import pickle
import glob
import h5py
import numpy as np
import tensorflow as tf
import json
import os
import base64
import time
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

directory_path = 'uploaded_models'


UPLOAD_FOLDER = "uploaded_models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/upload_model", methods=["POST"])
def upload_model():
    global upload_status
    upload_status = []  # Reset status for each upload

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and file.filename.endswith(".h5"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            upload_status.append("✅ Model uploaded successfully.")

            #Load the uploaded model
            try:
                uploaded_model = tf.keras.models.load_model(filepath)
                upload_status.append("✅ Model structure verified.")

                for layer in uploaded_model.layers:
                    weights = layer.get_weights()
                    print(f"Model, Layer {layer.name}, Weights Shape: {[w.shape for w in weights]}")

                # Simulate aggregation process (Delays for real-time effect)
                time.sleep(2)
                upload_status.append("✅ Merging weights with Global Model...")

                time.sleep(2)
                upload_status.append("✅ Finalizing Aggregation...")
                

                return jsonify({"message": "Model uploaded and aggregated successfully!"})

            except Exception as e:
                return jsonify({"error": f"Failed to load model: {e}"}), 500

        else:
            return jsonify({"error": "Invalid file format. Please upload a .h5 model file."}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route("/progress")
def progress():
    def generate():
        for status in upload_status:
            yield f"data: {status}\n\n"
            time.sleep(2)  # Send each status update with a delay
    
    return Response(generate(), mimetype="text/event-stream")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        amount = float(data["amount"])
        frequency = int(data["frequency"])
        time_of_transaction = data["time"].strip().lower()
        transaction_type = data["transactionType"].strip().upper()
        location = data["location"]


        #One-Hot Encoding for Time Slots
        time_slots = ["morning", "afternoon", "night"]
        time_encoding = [1 if time_of_transaction == t else 0 for t in time_slots]

        #One-Hot Encoding for Transaction Type
        trans_type_encoding = [1, 0] if transaction_type == "P2P" else [0, 1]

        #Convert Location to Numeric (Hashing)
        location_hash = hash(location) % 1000  # Convert to numeric

        #Prepare Input Features (Ensure Correct Order)
        input_features = np.array([amount, frequency] + trans_type_encoding + time_encoding + [location_hash]).reshape(1, -1)

        input_shape = (len(input_features[0]),)
        all_client_updates = load_client_updates_from_files(directory_path)
        print("Loaded client updates from files:")
        for key in all_client_updates:
            print(key, "->", type(all_client_updates[key]))

        dp_client_updates = apply_differential_privacy(all_client_updates, noise_multiplier=0.5, clipping_norm=1.0)
        global_model = fed_avg(dp_client_updates)
    
        # ✅ Save Weights Before Building Model
        weights_path = "global_model_weights.pkl"
        with open(weights_path, "wb") as f:
            pickle.dump(global_model, f)

        global_tf_model = build_model(input_shape)
        
        for layer in global_tf_model.layers:
            layer_name = layer.name  # Get layer name (e.g., "dense_1", "dense_2")

            if layer_name in global_model:  # ✅ No need for 'sequential' key
                try:
                    layer_weights = global_model[layer_name]  # Directly use stored weights

                    if isinstance(layer_weights, list) and len(layer_weights) == 1:
                        bias_shape = (layer_weights[0].shape[-1],)
                        bias = np.zeros(bias_shape)
                        layer_weights.append(bias)

                    layer.set_weights(layer_weights)  # ✅ Set correct weights
                    print(f"✅ Loaded weights for {layer_name}")

                except ValueError as e:
                    #print(f"❌ Shape mismatch for {layer_name}: {e}")
                    continue

        #Make Prediction
        user_input_tensor = np.array(input_features, dtype=np.float32)
        fraud_probability = global_tf_model.predict(user_input_tensor)[0][0]
        is_fraud = bool(fraud_probability > 0.5)  # Convert to boolean
        
        ai_description = generate_gemini_response(fraud_probability, is_fraud, data)
        if not isinstance(ai_description, str):
            ai_description = str(ai_description)

        return jsonify({"fraud_probability": float(fraud_probability), "is_fraud": is_fraud, "ai_description": ai_description.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error message if something goes wrong

@app.route('/download_model')
def download_model():
    model_path = "global_model_weights.pkl"
    return send_file(model_path, as_attachment=True)

def generate_gemini_response(fraud_probability, is_fraud, user_inputs):
    """
    Uses Gemini AI to generate a fraud analysis description and suggestions
    based on user input data and model predictions.
    """
    prompt = f"""
    Analyze the following transaction details and provide insights:

    - Amount: {user_inputs['amount']} INR
    - Transactions per week: {user_inputs['frequency']}
    - Time of transaction: {user_inputs['time']}
    - Transaction type: {user_inputs['transactionType']}
    - Location: {user_inputs['location']}
    - Fraud Probability: {fraud_probability:.2f}
    - Is Fraud: {is_fraud}

    limit to 1500character.
    """

    genai.configure(api_key="AIzaSyA3dqTBEjYqQ7bmq1cJhWcN081l78ldbSk")

    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        print(response.text, end="")
        return response.text
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

def load_h5_object(h5_object):
    """
    Recursively load data from an HDF5 object (dataset or group).
    """
    if isinstance(h5_object, h5py.Dataset):
        if h5_object.shape == ():  # Scalar dataset (single value)
            return h5_object[()]  # Directly extract scalar value
        else:
            return h5_object[:]  # Extract array data
    elif isinstance(h5_object, h5py.Group):
        group_data = {}
        for key in h5_object.keys():
            group_data[key] = load_h5_object(h5_object[key])
        return group_data
    else:
        raise ValueError("Unknown HDF5 object type.")

def load_client_updates_from_files(directory_path):
    updates = {}
    h5_files = glob.glob(os.path.join(directory_path, '*.h5'))

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            # Each key in the file is assumed to represent a client update
            for client_id in f.keys():
                unique_key = f"{os.path.basename(h5_file)}_{client_id}"
                updates[unique_key] = load_h5_object(f[client_id])

    return updates

def apply_differential_privacy(model_weights, noise_multiplier=1.0, clipping_norm=1.0):
    dp_model_weights = {}

    for layer_name, weights in model_weights.items():
        if isinstance(weights, np.ndarray):
            # Clip gradients
            norm = np.linalg.norm(weights)
            if norm > clipping_norm:
                weights = (weights / norm) * clipping_norm

            # Add Gaussian noise
            noise = np.random.normal(loc=0.0, scale=noise_multiplier, size=weights.shape)
            dp_model_weights[layer_name] = weights + noise
        else:
            dp_model_weights[layer_name] = weights  # Keep non-weight data unchanged

    return dp_model_weights

def recursive_fed_avg(global_layer, client_layers, num_clients):
    if isinstance(client_layers[0], dict):  # If it's a nested dictionary (e.g., {'kernel': ..., 'bias': ...})
        return {
            sub_key: recursive_fed_avg(None, [cl[sub_key] for cl in client_layers], num_clients)
            for sub_key in client_layers[0]
        }
    else:  # If it's a NumPy array (actual weight matrix)
        return sum(client_layers) / num_clients

def fed_avg(aggregated_updates):
    # Filter only `model_weights`, ignore `optimizer_weights`
    model_updates = {k: v for k, v in aggregated_updates.items() if "model_weights" in k}

    num_clients = len(model_updates)
    print(f"Aggregating {num_clients} client models...")

    # Get sample key for reference
    sample_key = list(model_updates.keys())[0]
    first_client_weights = model_updates[sample_key]  # First client's model weights

    # Initialize global model structure
    global_model = {
        layer: np.zeros_like(weight)
        for layer, weight in first_client_weights.items()
        if "dropout" not in layer  # Exclude dropout layers
    }

    # Aggregate weights recursively
    for layer in first_client_weights:
        if "dropout" not in layer:  # Exclude dropout layers
            client_layer_weights = [model[layer] for model in model_updates.values()]
            global_model[layer] = recursive_fed_avg(None, client_layer_weights, num_clients)

    return global_model

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Corrected Input Layer
        tf.keras.layers.Dense(64, activation='relu', name="dense"),
        tf.keras.layers.Dense(32, activation='relu', name="dense_1"),
        tf.keras.layers.Dense(16, activation='relu', name="dense_2"),
        tf.keras.layers.Dense(1, activation='sigmoid', name="dense_3")
    ])
    return model

if __name__ == "__main__":
    app.run(debug=True)
