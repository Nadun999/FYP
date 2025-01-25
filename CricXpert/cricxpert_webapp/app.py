from flask import Flask, request, jsonify, render_template
from inference.pipeline import predict_person, load_yolo  # Import your inference function
import os
import joblib
from keras.applications.resnet50 import ResNet50

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load required models and encoders
print("Loading models...")
net, output_layers = load_yolo()
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
ensemble_model = joblib.load('saved_models/ResNet/ensemble_player_recognition.pkl')
label_encoder = joblib.load('saved_models/ResNet/ensemble_label_encoder.pkl')
face_recognition_model = joblib.load('saved_models/Face_Recognition_Model/face_recognition_model.pkl')
face_label_encoder = joblib.load('saved_models/Face_Recognition_Model/label_encoder.pkl')
print("Models loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')  # Render the homepage

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Run inference
    try:
        result = predict_person(
            video_path=file_path,
            resnet_model=resnet_model,
            ensemble_model=ensemble_model,
            label_encoder=label_encoder,
            face_recognition_model=face_recognition_model,
            face_label_encoder=face_label_encoder,
        )
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)