from flask import Flask, render_template, request
import numpy as np
import pickle
from datetime import datetime
import requests
import os

app = Flask(__name__)

MODEL_PATH = "flight_delay_model.pkl"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/iu2uon64thsixcpprtp42/flight_delay_model.pkl?rlkey=um626etvhfu9e7poy2mc3ethq&st=4mwhsvo0&dl=1"

model = None

def load_model():
    global model
    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Dropbox...")
        response = requests.get(DROPBOX_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded.")
        else:
            raise Exception(f"Failed to download model: {response.status_code}")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("Model loaded.")
    return model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = load_model()  # Load only when needed

        flight_date = request.form["flight_date"]
        carrier = int(request.form["carrier"])
        origin = int(request.form["origin"])
        destination = int(request.form["destination"])
        crs_dep_time = int(request.form["crs_dep_time"])

        flight_datetime = datetime.strptime(flight_date, "%Y-%m-%d")
        month = flight_datetime.month - 1
        day = flight_datetime.day - 1
        weekday = flight_datetime.isoweekday() - 1

        features = np.array([[month, day, weekday, carrier, origin, destination, crs_dep_time, 0]])
        prediction = model.predict(features)[0]
        prediction_minutes = max(round(prediction, 2), 0)

        if prediction_minutes < 15:
            delay_category = "Low"
        elif prediction_minutes < 45:
            delay_category = "Medium"
        else:
            delay_category = "High"

        probability = min(int((prediction_minutes / 60) * 100), 100)
        factors = ["Weather", "Air Traffic", "Airline History"]

        return render_template("index.html",
                               prediction_text=f"Estimated delay: {prediction_minutes} minutes",
                               delay_category=delay_category,
                               probability=probability,
                               timestamp=datetime.now().strftime("%I:%M:%S %p"),
                               contributing_factors=factors)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
