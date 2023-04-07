from flask import Flask, request, render_template
from keras.models import load_model

from deepface import DeepFace
import cv2 as cv
import numpy as np
import base64

app = Flask(__name__)

model =load_model("Emotion.h5")
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_data = request.form['image']
        image_bytes = base64.b64decode(image_data)
        image_array = cv.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        try:
            gray = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Loop through each face detected in the frame
            for (x, y, w, h) in faces:
                # Extract the face ROI
                face_roi = gray[y:y + h, x:x + w]

                # Resize the face ROI to match the input size of the emotion detection model
                face_roi = cv.resize(face_roi, (48, 48))

                # Normalize the face ROI pixel values to be between 0 and 1
                face_roi = face_roi / 255.0

                # Reshape the face ROI to have a single channel (i.e., grayscale)
                face_roi = face_roi.reshape(1, 48, 48, 1)

                # Predict the emotion using the pre-trained model
                predictions = model.predict(face_roi)

                # Get the index of the predicted emotion (0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral)
                emotion_index = predictions.argmax()

            detected_emotion = emotions[emotion_index]
        except:
            detected_emotion = 'No Face is detected'
        return {'sentiment': detected_emotion}
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
