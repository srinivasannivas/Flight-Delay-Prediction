from flask import Flask, render_template, request
import numpy as np
import pickle
from datetime import datetime
import requests

app = Flask(__name__)

# Load model from Dropbox
DROPBOX_URL = "https://www.dropbox.com/scl/fi/iu2uon64thsixcpprtp42/flight_delay_model.pkl?rlkey=um626etvhfu9e7poy2mc3ethq&st=4mwhsvo0&dl=1"

def load_model():
    try:
        response = requests.get(DROPBOX_URL)
        response.raise_for_status()
        return pickle.loads(response.content)
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")

model = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        flight_date = request.form["flight_date"]
        carrier = int(request.form["carrier"])
        origin = int(request.form["origin"])
        destination = int(request.form["destination"])
        crs_dep_time = int(request.form["crs_dep_time"])

        # Extract date features
        flight_datetime = datetime.strptime(flight_date, "%Y-%m-%d")
        month = flight_datetime.month - 1
        day = flight_datetime.day - 1
        weekday = flight_datetime.isoweekday() - 1

        # Prepare features
        features = np.array([[month, day, weekday, carrier, origin, destination, crs_dep_time, 0]])
        prediction = model.predict(features)[0]
        prediction_minutes = max(round(prediction, 2), 0)

        # Determine category
        if prediction_minutes < 15:
            delay_category = "Low"
        elif prediction_minutes < 45:
            delay_category = "Medium"
        else:
            delay_category = "High"

        # Calculate probability
        probability = min(int((prediction_minutes / 60) * 100), 100)

        # Contributing factors (mocked)
        factors = ["Weather", "Air Traffic", "Airline History"]

        return render_template("index.html",
                               prediction_text=f"Estimated delay: {prediction_minutes} minutes",
                               delay_category=delay_category,
                               probability=probability,
                               timestamp=datetime.now().strftime("%I:%M:%S %p"),
                               contributing_factors=factors)

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

# Only for local development; for production, use gunicorn or other WSGI server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
