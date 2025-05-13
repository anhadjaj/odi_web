from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
import os

# Load scalers
with open("scaler_first.pkl", "rb") as f:
    scaler1 = pickle.load(f)

with open("scaler_second.pkl", "rb") as f:
    scaler2 = pickle.load(f)

# Load TensorFlow Lite interpreters
interpreter1 = tf.lite.Interpreter(model_path="odi_target_predictor_lstm.tflite")
interpreter1.allocate_tensors()
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()

interpreter2 = tf.lite.Interpreter(model_path="odi_chase_predictor_new.tflite")
interpreter2.allocate_tensors()
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()

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
        user_input_scaled = scaler1.transform(user_input).reshape(1, 1, 3).astype(np.float32)

        interpreter1.set_tensor(input_details1[0]['index'], user_input_scaled)
        interpreter1.invoke()
        prediction = interpreter1.get_tensor(output_details1[0]['index'])[0][0]

        return jsonify({"prediction": int(prediction)})
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
        user_input_scaled = scaler2.transform(user_input).astype(np.float32)

        interpreter2.set_tensor(input_details2[0]['index'], user_input_scaled)
        interpreter2.invoke()
        win_prob_team2 = interpreter2.get_tensor(output_details2[0]['index'])[0][0]
        win_prob_team1 = 1 - win_prob_team2
        probable_winner = "Team 2" if win_prob_team2 > win_prob_team1 else "Team 1"

        return jsonify({
            "Probability of Team 2 Winning": float(round(win_prob_team2 * 100, 2)),
            "Probability of Team 1 Winning": float(round(win_prob_team1 * 100, 2)),
            "Probable Winner": probable_winner
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)
    
if __name__ == '__main__':
    app.run(debug=True)