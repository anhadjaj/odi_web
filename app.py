from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pickle
import os
import gdown

# Google Drive file IDs (replace with your actual IDs)
model1_file_id = "1DS4kTPbhAsVh0UizYL9z6H2EaqZn9Hue"
model2_file_id = "1U0hqDYdFlqAOZhkedHlTLw-NG-cci23W"

# File names
model1_path = "odi_target_predictor_lstm.h5"
model2_path = "odi_chase_predictor_new.h5"

# Download models if not already present
if not os.path.exists(model1_path):
    url1 = f"https://drive.google.com/uc?id={model1_file_id}"
    gdown.download(url1, model1_path, quiet=False)

if not os.path.exists(model2_path):
    url2 = f"https://drive.google.com/uc?id={model2_file_id}"
    gdown.download(url2, model2_path, quiet=False)

# Load models
model1 = load_model(model1_path, custom_objects={"mse": MeanSquaredError()})
model2 = load_model(model2_path)

# Load pre-fitted scalers
with open("scaler_first.pkl", "rb") as f:
    scaler1 = pickle.load(f)

with open("scaler_second.pkl", "rb") as f:
    scaler2 = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_first_innings', methods=['POST'])
def predict_first_innings():
    try:
        input_data = request.json
        curr_runs = float(input_data["curr_runs"])
        curr_wickets = float(input_data["curr_wickets"])
        overs = float(input_data["overs"])

        user_input = np.array([[curr_runs, curr_wickets, overs]])
        user_input_scaled = scaler1.transform(user_input).reshape(1, 1, 3)

        prediction = model1.predict(user_input_scaled)[0][0]

        return jsonify({"prediction": int(prediction.item())})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_second_innings', methods=['POST'])
def predict_second_innings():
    try:
        input_data = request.json
        target = float(input_data['target'])
        curr_runs = float(input_data['curr_runs'])
        curr_wickets = float(input_data['curr_wickets'])
        overs = float(input_data['overs'])

        user_input = np.array([[target, curr_runs, curr_wickets, overs]])
        user_input_scaled = scaler2.transform(user_input)

        win_prob_team2 = model2.predict(user_input_scaled)[0][0]
        win_prob_team1 = 1 - win_prob_team2
        probable_winner = "Team 2" if win_prob_team2 > win_prob_team1 else "Team 1"

        return jsonify({
            "Probability of Team 2 Winning": round(float(win_prob_team2) * 100, 2),
            "Probability of Team 1 Winning": round(float(win_prob_team1) * 100, 2),
            "Probable Winner": probable_winner
        })


    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
