import cv2
import os
import pickle
import pandas as pd
import datetime
from PIL import Image
from keras.models import load_model  # Assuming FaceNet model and dependencies are installed
from keras_facenet import FaceNet

# Function to pre-process and extract face embeddings
def extract_face(image_path):
    try:
        # Load image and convert to RGB
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Haar cascade classifier to detect faces
        haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = haarcascade.detectMultiScale(img, 1.1, 4)

        if len(faces) == 0:
            print("No face detected in the image:", image_path)
            return None

        # Extract the first detected face (assuming single face)
        x1, y1, width, height = faces[0]
        face = img[y1:y1+height, x1:x1+width]

        # Convert PIL image to array, resize, and expand dimensions
        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)
        face = expand_dims(face, axis=0)

        # Calculate face signature using FaceNet
        face_net = FaceNet()
        signature = face_net.embeddings(face)

        return signature
    except Exception as e:
        print("Error processing image:", image_path, e)
        return None

# Load or create the attendance DataFrame
excel_file_path = 'face_data.xlsx'
try:
    df = pd.read_excel(excel_file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
    df.to_excel(excel_file_path, index=False)

# Load the face signatures data
with open("data.pkl", "rb") as myFile:
    database = pickle.load(myFile)

# Initialize video capture and window
cap = cv2.VideoCapture(0)
cv2.namedWindow('Face Attendance System', cv2.WINDOW_NORMAL)

# Display window to ensure it's not minimized
_, gbr1 = cap.read()
cv2.imshow('Face Attendance System', gbr1)
cv2.waitKey(1)  # Bring window to focus

while True:
    ret, frame = cap.read()

    # Process frame to detect and recognize face
    signature = extract_face(frame)
    if signature is not None:
        min_dist = 100
        identity = 'Unknown'

        for key, value in database.items():
            dist = np.linalg.norm(value - signature)
            if dist < min_dist:
                min_dist = dist
                identity = key

        # Get current time and date
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Check if the name already exists for the current date
        if df.empty or not df[(df['Name'] == identity) & (df['Date'] == current_date)].empty:
            # Create a new entry
            new_entry = pd.DataFrame({'Name': [identity], 'Time': [current_time], 'Date': [current_date]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_excel(excel_file_path, index=False)

            print(f"Attendance recorded for {identity} at {current_time} on {current_date}")

        # Display recognized name and bounding box
        cv2.putText(frame, identity, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Show face rectangle if applicable

    # Display the frame
    cv2.imshow('Face Attendance System', frame)

    # Check for window closure or "q" key press
    k = cv2.waitKey(1) & 0
