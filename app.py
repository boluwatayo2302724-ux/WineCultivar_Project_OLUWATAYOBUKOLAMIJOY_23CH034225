from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


# Load model and scaler
model = joblib.load('model/wine_cultivar_model.pkl')
scaler = joblib.load('model/scaler.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        features = [
            float(request.form['alcohol']),
            float(request.form['malic_acid']),
            float(request.form['alcalinity_of_ash']),
            float(request.form['magnesium']),
            float(request.form['color_intensity']),
            float(request.form['proline'])
        ]


        scaled_features = scaler.transform([features])
        pred_class = model.predict(scaled_features)[0]


        prediction = f"Cultivar {pred_class + 1}" # Convert 0,1,2 → 1,2,3


    return render_template('index.html', prediction=prediction)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

