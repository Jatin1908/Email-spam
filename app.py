from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form.get('message')

    # Safety check
    if not msg:
        return render_template("index.html", result="⚠️ Please enter a message")

    # Transform input
    data = vectorizer.transform([msg])

    # Predict
    prediction = model.predict(data)[0]

    # Result + status for UI color
    if prediction == 1:
        result = "🚨 Spam Message"
        status = "spam"
    else:
        result = "✅ Not Spam"
        status = "ham"

    return render_template("index.html", result=result, status=status)

# Run app
if __name__ == "__main__":
    app.run(debug=True)