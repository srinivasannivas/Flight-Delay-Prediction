import joblib

# Load your encoders
encoders = joblib.load("encoders.pkl")

# Print valid label values for each feature
for name, enc in encoders.items():
    print(f"{name}: {list(enc.classes_)}")
