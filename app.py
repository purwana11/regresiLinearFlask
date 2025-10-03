from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Dapatkan direktori saat ini
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_student_bundle.pkl")

# Load model bundle
bundle = joblib.load(model_path)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Ambil input dari form
            hours = float(request.form["hours"])
            prev_scores = float(request.form["prev_scores"])
            extracurricular = int(request.form["extracurricular"])  # 1=Yes, 0=No
            sleep_hours = float(request.form["sleep_hours"])
            papers = float(request.form["papers"])

            # Susun array sesuai urutan fitur
            data = np.array([[hours, prev_scores, extracurricular, sleep_hours, papers]])

            # Prediksi
            prediction = model.predict(data)[0]
            prediction = max(0, min(100, prediction))
            prediction = round(float(prediction), 2)

            return render_template("index.html", 
                                   result=f"Prediksi Performance Index: {prediction}")
        except Exception as e:
            return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
