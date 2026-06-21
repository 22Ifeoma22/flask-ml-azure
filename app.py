from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Iris Prediction API Running"

@app.route("/predict")
def predict():
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    return f"Prediction: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True)
