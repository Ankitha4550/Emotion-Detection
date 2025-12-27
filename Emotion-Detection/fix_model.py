import tensorflow as tf
from tensorflow.keras.models import load_model

# Load old model using TF (legacy compatible)
model = load_model(
    r"C:\Users\ankit\OneDrive\Desktop\deeee\Emotion-Detection\Emotion_Detection.h5",
    compile=False
)

# Save again in modern Keras format
model.save(
    r"C:\Users\ankit\OneDrive\Desktop\deeee\Emotion-Detection\Emotion_Detection.keras"
)

print("✅ Model converted successfully!")
