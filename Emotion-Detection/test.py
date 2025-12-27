import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
from tensorflow.keras.models import load_model

classifier = load_model(
    r"C:\Users\ankit\OneDrive\Desktop\deeee\Emotion-Detection\Emotion_Detection.keras"
)


# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\ankit\OneDrive\Desktop\deeee\Emotion-Detection\haarcascade_frontalface_default.xml"
)

# FER2013 emotion labels (IMPORTANT)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam (Windows fix)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # ✅ CORRECT SIZE
        roi = roi_gray.astype('float32') / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))       # ✅ CORRECT SHAPE

        preds = classifier.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
