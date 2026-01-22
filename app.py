from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("wine_cultivar_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["ash"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["flavanoids"]),
            float(request.form["color_intensity"])
        ]

        scaled_features = scaler.transform([features])
        result = model.predict(scaled_features)[0]

        prediction = f"Cultivar {result + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


