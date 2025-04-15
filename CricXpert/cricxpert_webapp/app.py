from flask import Flask, request, jsonify, render_template
from inference.pipeline import predict_person, load_yolo
from inference.stat_generation import generate_sql_query, player_stats
import os
import joblib
import logging

from keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import load_model
from langchain_openai import ChatOpenAI
from werkzeug.utils import secure_filename

app = Flask(__name__)

# configure logging
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading models...")
net, output_layers = load_yolo()
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
ensemble_model = joblib.load('saved_models/ResNet/ensemble_player_recognition.pkl')
label_encoder = joblib.load('saved_models/ResNet/ensemble_label_encoder.pkl')
face_recognition_model = joblib.load('saved_models/Face_Recognition_Model/face_recognition_model.pkl')
face_label_encoder = joblib.load('saved_models/Face_Recognition_Model/label_encoder.pkl')
temporal_model = load_model('saved_models/GRU/temporal_model')
llm = ChatOpenAI(model="gpt-4o", openai_api_key="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")

print("Models loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/cricxpert_webapp/uploads', filename)
        file.save(file_path)

        # determine if the file is a video or an image
        is_video = filename.lower().endswith(('.mp4', '.mov', '.avi'))
        logging.debug(f"File uploaded: {filename}, is_video: {is_video}")

        try:
            result = predict_person(
                video_path=file_path,
                resnet_model=resnet_model,
                ensemble_model=ensemble_model,
                label_encoder=label_encoder,
                face_recognition_model=face_recognition_model,
                face_label_encoder=face_label_encoder,
                temporal_model=temporal_model,
                is_video=is_video
            )
            logging.debug(f"Prediction result: {result}")
            return jsonify({"prediction": result})
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
@app.route('/get_player_stats', methods=['POST'])
def get_player_stats():
    data = request.get_json()
    player = data.get('player')
    if player in player_stats:
        return jsonify({"stats": player_stats[player]})
    else:
        return jsonify({"error": "Player not found"}), 404

@app.route('/generate_stat', methods=['POST'])
def generate_stat():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"})

    try:
        result = generate_sql_query(user_question, llm)
        return jsonify({"stat_result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)