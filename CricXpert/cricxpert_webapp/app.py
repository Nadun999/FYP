from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from inference.pipeline import predict_player  # Import your inference function

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded videos
UPLOAD_FOLDER = 'CricXpert/cricxpert_webapp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to render upload page
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run inference
        result = predict_player(filepath)
        
        # Return result as JSON (can be displayed on the frontend)
        return jsonify(result)
    else:
        return "Invalid file type. Only MP4/MOV/AVI are allowed.", 400

if __name__ == '__main__':
    app.run(debug=True)
