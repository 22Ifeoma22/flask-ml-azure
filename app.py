from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Iris Prediction API Running"

@app.route("/predict")
def predict():
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])[0]

    labels = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    return jsonify({
        "prediction_code": int(prediction),
        "prediction_name": labels[int(prediction)]
    })

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
