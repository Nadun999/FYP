import os
import cv2
import numpy as np
import pytesseract
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter
import mediapipe as mp

import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2


# path to tesseract 
pytesseract.pytesseract.tesseract_cmd = ('/opt/homebrew/bin/tesseract') 

# player database with both full reference name and jersey name
player_database = {
    "Virat_Kohli": {"name": "VIRAT", "number": "18"},
    "Arshdeep_Singh": {"name": "ARSHDEEP", "number": "2"},
    "Axar_Patel": {"name": "AXAR", "number": "20"},
    "Jasprit_Bumrah": {"name": "JASPRIT", "number": "93"},
    "Kuldeep_Yadav": {"name": "KULDEEP", "number": "23"},
    "Ravindra_Jadeja": {"name": "JADEJA", "number": "8"}
}

def load_yolo():
    path_to_cfg = "/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Hybrid_Spatio_Temporal_Model_For_Gait_Analysis/yolo/yolov3.cfg" 
    path_to_weights = "/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Hybrid_Spatio_Temporal_Model_For_Gait_Analysis/yolo/yolov3.weights"
    net = cv2.dnn.readNet(path_to_weights, path_to_cfg)
    layers_names = net.getLayerNames()
    
    try:
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers

def yolo_detect(net, image, output_layers, confidence_threshold=0.3):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and class_id == 0:  # player class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if x >= 0 and y >= 0 and (x + w) <= width and (y + h) <= height:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

    # select the largest vertical box based on area if any boxes were detected
    if boxes:
        largest_box = max(boxes, key=lambda b: b[2] * b[3])  # b[2]*b[3] is the area of the box (w*h)
        largest_confidence = confidences[boxes.index(largest_box)]
        return [largest_box], [largest_confidence]
    return [], []  # return empty lists if no boxes detected


def extract_features(images, model):
    processed_images = []

    for img in images:
        if img is not None and img.size > 0:  # to ensure the image is not empty
            resized_img = cv2.resize(img, (224, 224))
            processed_images.append(resized_img)
    
    if not processed_images:
        return np.array([])  # return empty array if no images to process
    
    images_array = np.array(processed_images)
    images_array = preprocess_input(images_array)
    features = model.predict(images_array)
    
    return features

    # features = features.reshape((features.shape[0], -1))


def process_frame_for_OCR_text_detection(image):

    # convert image to RGB for consistent display if originally in BGR
    if image.shape[2] == 3:  # assuming the image has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # original dimensions
    (H, W) = image.shape[:2]

    # set the new width and height to nearest multiple of 32 for EAST model
    newW = int(W / 32) * 32
    newH = int(H / 32) * 32

    # resize the image to fit model requirements
    image = cv2.resize(image, (newW, newH))

    # load the pre-trained EAST text detector model
    model_path = '/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Hybrid_Spatio_Temporal_Model_For_Gait_Analysis/saved_models/ResNet/frozen_east_text_detection.pb'
    net = cv2.dnn.readNet(model_path)

    # prepare the image for the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # layer names for the output layers
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # forward pass of the model to get output
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to avoid overlaps
    indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

    # # Plot the figures side by side
    # plt.figure(figsize=(18, 12))


    # # Plot original detections
    # plt.subplot(1, 3, 1)
    # plt.imshow(image)
    # for i in indices.flatten():
    #     (startX, startY, endX, endY) = rects[i]
    #     rect = plt.Rectangle((startX, startY), endX - startX, endY - startY, edgecolor='r', facecolor='none')
    #     plt.gca().add_patch(rect)
    # plt.title('Original Detections')
    # plt.axis('off')


    # # Plot original detections
    # plt.subplot(1, 3, 1)
    # plt.imshow(image)

    if len(indices) > 0:
        indices = indices.flatten()  # ensuring flattening is possible

        # # Plotting and processing detections
        # plt.figure(figsize=(10, 6))

        # for i in indices.flatten():
            # (startX, startY, endX, endY) = rects[i]
            # rect = plt.Rectangle((startX, startY), endX - startX, endY - startY, edgecolor='r', facecolor='none')
            # plt.gca().add_patch(rect)
    # plt.title('Original Detections')
    # plt.axis('off')

    # # Plot merged box
    # plt.subplot(1, 3, 2)
    # plt.imshow(image)

    cropped_img = None

    if len(indices) > 0:
        min_x = max(0, min([rects[i][0] for i in indices]) - 20)
        min_y = max(0, min([rects[i][1] for i in indices]) - 20)
        max_x = min(W, max([rects[i][2] for i in indices]) + 20)
        max_y = min(H, max([rects[i][3] for i in indices]) + 20)

        cropped_img = image[min_y:max_y, min_x:max_x]

        # merged_rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, edgecolor='r', facecolor='none')
        # plt.gca().add_patch(merged_rect)
        # plt.title('Merged Detections with Expanded Box')
    else:
        print('No Detections Found')
    # plt.axis('off')

    # # Display the cropped area from the merged box
    # plt.subplot(1, 3, 3)
    # if len(indices) > 0:
    #     plt.imshow(cropped_img)
    #     plt.title('Cropped Area from Merged Box')
    # else:
    #     plt.title('No Area to Crop')
    # plt.axis('off')

    # plt.show()


    # Initialize 'text' to an empty string
    text = ''

    # proceed with text recognition on the cropped image
    if cropped_img is not None:
        # convert cropped image from RGB to BGR for OpenCV operations
        cropped_img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        
        # convert to grayscale
        gray = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2GRAY)

        # # Apply thresholding
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # use median blur to remove noise
        blur = cv2.medianBlur(gray, 5)

        # # Resize for better accuracy
        # resized_img = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Configure parameters for Tesseract
        custom_config = r'--oem 3 --psm 11'
        text = pytesseract.image_to_string(blur, config=custom_config)

        # # plotting the preprocessing visualization of the cropped image
        # plt.figure(figsize=(12, 10))
        # plt.subplot(1, 3, 1)
        # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        # plt.title('Grayscale')

        # plt.subplot(1, 3, 2)
        # plt.imshow(blur, cmap='gray')
        # plt.title('Median Blur')

        # plt.subplot(1, 3, 3)
        # plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        # plt.title('Resized for OCR')

        # plt.show()

        print("Detected text:", text)
    else:
        print("No area was cropped for text recognition.")

    return text


# load the FaceNet model for embeddings
embedder = FaceNet()

# initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# function to get embeddings using FaceNet
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

# function to extract pose landmarks from the cropped image
def extract_pose_landmarks(cropped_img):
    # convert the image to RGB
    image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    # perform pose detection
    results = pose.process(image_rgb)

    # if landmarks are detected
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).flatten()
    return None

# calculate step length between left and right ankles
def calculate_step_length(pose_sequence):
    left_ankle_x, left_ankle_y = pose_sequence[27*3], pose_sequence[27*3 + 1]  # left ankle
    right_ankle_x, right_ankle_y = pose_sequence[28*3], pose_sequence[28*3 + 1]  # right ankle
    step_length = np.linalg.norm(np.array([right_ankle_x, right_ankle_y]) - np.array([left_ankle_x, left_ankle_y]))
    return step_length

# calculate joint velocities between two consecutive frames
def calculate_velocity(pose_sequence_t, pose_sequence_t1, joint_index):
    joint_t = np.array([pose_sequence_t[joint_index*3], pose_sequence_t[joint_index*3 + 1]])
    joint_t1 = np.array([pose_sequence_t1[joint_index*3], pose_sequence_t1[joint_index*3 + 1]])
    velocity = np.linalg.norm(joint_t1 - joint_t)
    return velocity

# calculate joint angles (hip-knee-ankle)
def calculate_joint_angle(pose_sequence, hip_idx, knee_idx, ankle_idx):
    hip = np.array([pose_sequence[hip_idx*3], pose_sequence[hip_idx*3 + 1]])
    knee = np.array([pose_sequence[knee_idx*3], pose_sequence[knee_idx*3 + 1]])
    ankle = np.array([pose_sequence[ankle_idx*3], pose_sequence[ankle_idx*3 + 1]])
    
    # calculate vectors
    vec_hip_knee = knee - hip
    vec_knee_ankle = ankle - knee
    
    # calculate the cosine of the angle between the vectors
    cos_angle = np.dot(vec_hip_knee, vec_knee_ankle) / (np.linalg.norm(vec_hip_knee) * np.linalg.norm(vec_knee_ankle))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

# calculate joint accelerations between two consecutive frames (acceleration is the change in velocity)
def calculate_acceleration(velocity_t, velocity_t1):
    acceleration = velocity_t1 - velocity_t
    return acceleration

# calculate angular velocity (rate of change of joint angles between consecutive frames)
def calculate_angular_velocity(joint_angle_t, joint_angle_t1):
    angular_velocity = joint_angle_t1 - joint_angle_t
    return angular_velocity

# calculate hip displacement between two frames
def calculate_hip_displacement(pose_sequence_t, pose_sequence_t1):
    left_hip_t = np.array([pose_sequence_t[23*3], pose_sequence_t[23*3 + 1]])
    right_hip_t = np.array([pose_sequence_t[24*3], pose_sequence_t[24*3 + 1]])
    left_hip_t1 = np.array([pose_sequence_t1[23*3], pose_sequence_t1[23*3 + 1]])
    right_hip_t1 = np.array([pose_sequence_t1[24*3], pose_sequence_t1[24*3 + 1]])
    
    # calculate hip centers
    hip_center_t = (left_hip_t + right_hip_t) / 2
    hip_center_t1 = (left_hip_t1 + right_hip_t1) / 2
    
    # displacement between frames
    hip_displacement = np.linalg.norm(hip_center_t1 - hip_center_t)
    return hip_displacement


# temporal feature extraction (step length, joint velocities, joint angles, etc.)
def calculate_features(pose_landmarks, previous_landmarks=None, previous_velocities=None, previous_angles=None):
    frame_features = [0] * 10  # 10 features (step length, velocity x2, joint angles x2, acceleration x2, angular velocity x2, hip displacement)

    # step Length
    step_length = calculate_step_length(pose_landmarks)
    frame_features[0] = step_length

    # joint Velocities
    if previous_landmarks is not None:
        left_ankle_velocity = calculate_velocity(previous_landmarks, pose_landmarks, 27)
        right_ankle_velocity = calculate_velocity(previous_landmarks, pose_landmarks, 28)
        frame_features[1] = left_ankle_velocity
        frame_features[2] = right_ankle_velocity

        # joint Accelerations
        if previous_velocities is not None:
            left_ankle_acceleration = calculate_acceleration(previous_velocities[0], left_ankle_velocity)
            right_ankle_acceleration = calculate_acceleration(previous_velocities[1], right_ankle_velocity)
            frame_features[5] = left_ankle_acceleration
            frame_features[6] = right_ankle_acceleration

        # hip Displacement
        hip_displacement = calculate_hip_displacement(previous_landmarks, pose_landmarks)
        frame_features[9] = hip_displacement
    else:
        frame_features[1] = frame_features[2] = 0  # velocities
        frame_features[5] = frame_features[6] = 0  # accelerations
        frame_features[9] = 0  # hip displacement

    # joint angles
    left_leg_angle = calculate_joint_angle(pose_landmarks, 23, 25, 27)
    right_leg_angle = calculate_joint_angle(pose_landmarks, 24, 26, 28)
    frame_features[3] = left_leg_angle
    frame_features[4] = right_leg_angle

    # angular velocities
    if previous_angles is not None:
        left_leg_angular_velocity = calculate_angular_velocity(previous_angles[0], left_leg_angle)
        right_leg_angular_velocity = calculate_angular_velocity(previous_angles[1], right_leg_angle)
        frame_features[7] = left_leg_angular_velocity
        frame_features[8] = right_leg_angular_velocity
    else:
        frame_features[7] = frame_features[8] = 0  # angular velocities

    return frame_features

# extract temporal features from frames
def extract_temporal_features(frames):
    temporal_features = []
    previous_landmarks = None
    previous_velocities = None
    previous_angles = None

    for frame in frames:
        pose_landmarks = extract_pose_landmarks(frame)
        if pose_landmarks is not None:
            frame_features = calculate_features(pose_landmarks, previous_landmarks, previous_velocities, previous_angles)
            temporal_features.append(frame_features)
            previous_landmarks = pose_landmarks  # store the current landmarks for the next frame

    return np.array(temporal_features)

def clean_detected_text(text):
    # remove non-alphanumeric characters and extra spaces
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = text.strip().upper()  # convert to uppercase for case-insensitive matching

    return text

def extract_frames_for_prediction(video_path, net, output_layers, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames if total_frames > num_frames else 1
    frame_ids = [int(interval * i) for i in range(num_frames)]
    frames = []
    detected_texts = [] 

    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_ids:
            # apply YOLO detection to each frame
            boxes, confidences = yolo_detect(net, frame, output_layers)
            if boxes:  # check if there is at least one detection
                largest_box = boxes[0]  # the largest box returned by yolo_detect
                x, y, w, h = largest_box  # unpack the largest box
                cropped_frame = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                resized_frame = cv2.resize(cropped_frame, (224, 224))  # resize frame

                # apply CLAHE for better contrast
                lab = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                processed_img = cv2.merge([l, a, b])
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_LAB2BGR)

                frames.append(processed_img)
                detected_text = process_frame_for_OCR_text_detection(cropped_frame)
                detected_texts.append(clean_detected_text(detected_text))  # clean and append OCR text
            else:
                # fallback for frames without valid detections
                resized_frame = cv2.resize(frame, (224, 224))  # resize original frame
                frames.append(resized_frame)  # include the resized frame
                print(f"No valid detections at frame {frame_count}.")
        frame_count += 1

    cap.release()
    return frames, detected_texts

def extract_image_for_prediction(image_path, net, output_layers):
    frame = cv2.imread(image_path)
    frames = []
    detected_texts = []

    # apply YOLO detection to the image
    boxes, confidences = yolo_detect(net, frame, output_layers)
    if boxes:  # check if there is at least one detection
        largest_box = boxes[0]  #the largest box returned by yolo_detect
        x, y, w, h = largest_box  # unpack the largest box
        cropped_frame = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        resized_frame = cv2.resize(cropped_frame, (224, 224))  # resize frame

        # apply CLAHE for better contrast
        lab = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        processed_img = cv2.merge([l, a, b])
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_LAB2BGR)

        frames.append(processed_img)
        detected_text = process_frame_for_OCR_text_detection(cropped_frame)
        detected_texts.append(clean_detected_text(detected_text))  # clean and append OCR text
    else:
        # fallback for images without valid detections
        resized_frame = cv2.resize(frame, (224, 224))  # resize original frame
        frames.append(resized_frame)  # include the resized frame
        print("No valid detections in the image.")

    return frames, detected_texts

def contains_substring(player_name, detected_text, min_length=4):
    player_name = player_name.upper()  # ensure case-insensitive matching
    detected_text = detected_text.upper()

    # sliding window approach: check all substrings of `player_name`
    for i in range(len(player_name) - min_length + 1):
        substring = player_name[i:i + min_length]
        if substring in detected_text:
            return True
    return False





# predict player using the entire pipeline (OCR, Face, Spatial, and Temporal models)
def predict_person(video_path, resnet_model, ensemble_model, label_encoder, face_recognition_model, face_label_encoder, temporal_model, is_video=True):

    # load YOLO net
    net, output_layers = load_yolo()
    # load the pre-trained ResNet model
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # load trained model and label encoder for the spatial model
    ensemble_model = joblib.load('/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Hybrid_Spatio_Temporal_Model_For_Gait_Analysis/saved_models/ResNet/ResNet_SVM_KNN/ensemble_player_recognition.pkl')
    label_encoder = joblib.load('/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Hybrid_Spatio_Temporal_Model_For_Gait_Analysis/saved_models/ResNet/ResNet_SVM_KNN/ensemble_label_encoder.pkl')
    # load face recognition model and label encoder
    face_recognition_model = joblib.load('/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Face_Recognition_Model/trained_model/face_recognition_model.pkl')
    face_label_encoder = joblib.load('/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Face_Recognition_Model/trained_model/label_encoder.pkl')
    # load the GRU-based temporal model
    temporal_model = load_model('/Users/nadunsenarathne/Downloads/Documents/IIT/4th Year/FYP/CricXpert/Hybrid_Spatio_Temporal_Model_For_Gait_Analysis/saved_models/GRU/temporal_model')


    if is_video:
        frames, detected_texts = extract_frames_for_prediction(video_path, net, output_layers)
    else:
        frames, detected_texts = extract_image_for_prediction(video_path, net, output_layers)

    if not frames:
        print("No frames to analyze or no valid detections.")
        return

    # initialize a list to store text-based matches
    text_matches = []

    # Step 1: OCR System (Try to detect jersey name/number)
    for i, detected_text in enumerate(detected_texts):
        detected_text_upper = detected_text.upper().strip()
        print(f"\nProcessing frame {i}")
        print(f"Detected text: {detected_text_upper}")
        matched_player = None
        for player, info in player_database.items():
            name_match = contains_substring(info['name'], detected_text_upper, min_length=4)
            number_match = info['number'] in detected_text_upper
            if name_match and number_match:
                matched_player = player
                print(f"Detected text matches player: {player}")
                text_matches.append(matched_player)
                break  # stop checking after finding a match
            elif name_match:
                matched_player = player
                print(f"Detected text matches player: {player}")
                text_matches.append(matched_player)
                break  # stop checking after finding a match

    # check if any text matches were found from OCR
    if text_matches:
        most_common_player = Counter(text_matches).most_common(1)[0][0]
        print(f"\nPredicted player based on text detection: {most_common_player}")
        return most_common_player

    print("No player detected from OCR, moving to face recognition...")


    # Step 2: Face Recognition (Use MTCNN and FaceNet)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames_to_extract = 10
    interval = total_frames // num_frames_to_extract if total_frames >= num_frames_to_extract else 1
    frame_ids = [interval * i for i in range(num_frames_to_extract)]
    frames = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_ids:
            # append raw frames without YOLO cropping
            frames.append(frame)

        frame_count += 1
    cap.release()

    # check if frames were extracted
    if not frames:
        print("No frames were extracted from the video.")
        return "Unknown"
    

    processed_faces = []
    face_predictions = []
    detector = MTCNN()

    for i, frame in enumerate(frames):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_img)
        
        if results:
            for result in results:
                x, y, w, h = result['box']
                x, y = max(0, x), max(0, y)  # ensure coordinates are within the image
                face = rgb_img[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                
                # generate embedding for the detected face
                embedding = get_embedding(face)
                embedding = np.expand_dims(embedding, axis=0)

                # predict the identity of the face
                ypred = face_recognition_model.predict(embedding)
                proba = face_recognition_model.predict_proba(embedding).max()
                if proba > 0.5:  # confidence threshold
                    final_name = face_label_encoder.inverse_transform(ypred)[0]
                    face_predictions.append(final_name)
                    break  # stop processing further faces in the frame
                else:
                    face_predictions.append("Unknown")
        else:
            print(f"No face detected in frame {i}.")

    # check if any face matches were found
    if face_predictions:
        filtered_face_predictions = [pred for pred in face_predictions if pred != "Unknown"]
        if filtered_face_predictions:
            most_common_face_prediction = Counter(filtered_face_predictions).most_common(1)[0][0]
            print(f"\nPredicted player based on face recognition: {most_common_face_prediction}")
            return most_common_face_prediction

    print("No player detected from face recognition, moving to spatial model...")

    # Step 3: Spatial Model (Use ResNet for feature extraction and ensemble model for prediction)
    features = extract_features(frames, resnet_model)
    predictions = []
    for i, feature in enumerate(features):
        probabilities = ensemble_model.predict_proba([feature])[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_player = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence = probabilities[predicted_class_index]
        print(f"Frame {i}: Detected person: {predicted_player} with confidence: {confidence:.2f}")

        if confidence > 0.1:
            predictions.append(predicted_player)
        else:
            predictions.append("Unknown")

    # check if any spatial model predictions were made
    if predictions:
        filtered_predictions = [pred for pred in predictions if pred != "Unknown"]
        if filtered_predictions:
            most_common_prediction = Counter(filtered_predictions).most_common(1)[0][0]
            print(f"\nPredicted player based on spatial model: {most_common_prediction}")
            return most_common_prediction

    print("No player detected from spatial model, moving to temporal model...")


    # Step 4: Temporal Model (Use pose landmarks and GRU-based model for gait analysis)
    temporal_features = extract_temporal_features(frames)
    sequence_length = 15

    # if len(temporal_features) < sequence_length:
    #     # Pad or duplicate the temporal features to match the sequence length
    #     padding_needed = sequence_length - len(temporal_features)
    #     padded_temporal_features = np.pad(
    #         temporal_features,
    #         ((0, padding_needed), (0, 0)),  # Pad rows and columns
    #         mode='edge'  # Duplicate the last row
    #     )
    #     temporal_features = padded_temporal_features

    # create sequences from temporal features
    temporal_sequences = []
    for i in range(len(temporal_features) - sequence_length + 1):
        temporal_sequences.append(temporal_features[i:i + sequence_length])
    temporal_sequences = np.array(temporal_sequences)

    # check if temporal_sequences is empty
    if len(temporal_sequences) == 0:
        print("Not enough temporal sequences for analysis. Returning 'Unknown'.")
        return "Unknown"

    # predict using the temporal model
    temporal_predictions = temporal_model.predict(temporal_sequences)

    # aggregate predictions to find the most common player
    predicted_classes = np.argmax(temporal_predictions, axis=1)
    class_counts = np.bincount(predicted_classes)

    # if no clear prediction can be made, return 'Unknown'
    if len(class_counts) == 0 or class_counts.max() == 0:
        print("No clear player prediction from temporal model. Returning 'Unknown'.")
        return "Unknown"

    predicted_player = np.argmax(class_counts)
    most_common_temporal_prediction = label_encoder.inverse_transform([predicted_player])[0]
    print(f"\nPredicted player based on temporal model: {most_common_temporal_prediction}")
    return most_common_temporal_prediction