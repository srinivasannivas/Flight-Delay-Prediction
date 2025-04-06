from flask import Flask, render_template, request
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)
model = joblib.load("flight_delay_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        flight_date = request.form["flight_date"]
        carrier = int(request.form["carrier"])
        origin = int(request.form["origin"])
        destination = int(request.form["destination"])
        crs_dep_time = int(request.form["crs_dep_time"])

        # Parse date
        flight_datetime = datetime.strptime(flight_date, "%Y-%m-%d")
        month = flight_datetime.month - 1
        day = flight_datetime.day - 1
        weekday = flight_datetime.isoweekday() - 1

        features = np.array([[month, day, weekday, carrier, origin, destination, crs_dep_time, 0]])
        prediction = model.predict(features)[0]
        prediction_minutes = max(round(prediction, 2), 0)

        # Delay category
        if prediction_minutes < 15:
            delay_category = "Low"
        elif prediction_minutes < 45:
            delay_category = "Medium"
        else:
            delay_category = "High"

        # Delay probability (mock calculation)
        probability = min(int((prediction_minutes / 60) * 100), 100)

        # Mock contributing factors
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
    app.run(debug=True)
