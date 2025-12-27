

# Emotion Detection Using CNN and OpenCV

## Overview

This project implements a **real-time emotion detection system** using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. The system can detect facial emotions such as **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral** using a webcam.

## Features

* Real-time emotion detection from webcam feed
* Supports 7 basic emotions
* Built with **Python**, **TensorFlow/Keras**, and **OpenCV**
* Cross-platform support (Windows tested)

## Dataset

* **FER-2013** (Facial Expression Recognition)
* Grayscale images, 48x48 pixels
* ~28,000 training images, ~3,500 test images
* 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Requirements

* Python 3.x
* TensorFlow
* Keras
* OpenCV
* NumPy

Install dependencies using pip:

 pip install tensorflow keras opencv-python numpy

## Usage

1. Clone the repository:

  git clone <your-repo-url>
  cd <your-repo-name>


2. Activate your virtual environment (if using `.venv`):

# Windows PowerShell
   & .\.venv\Scripts\Activate.ps1

3. Run the training script (optional if you want to retrain):

  python train.py

4. Run the real-time detection:

  python test.py

5. Press **'q'** to quit the webcam window.

## File Structure

Emotion-Detection/
│
├─ train.py                 # Training script for CNN model
├─ test.py                  # Real-time detection script
├─ fix_model.py             # Script for fixing model issues
├─ Emotion_Detection.keras  # Trained model (Keras format)
├─ haarcascade_frontalface_default.xml
├─ README.md

## Results

* Validation Accuracy: ~45-50% on FER-2013
* Real-time webcam detection works for single or multiple faces
* Example screenshot of detected emotions: [Add a screenshot here]

## Challenges & Solutions

| Challenge            | Solution                                  |
| -------------------- | ----------------------------------------- |
| `batch_shape` error  | Re-saved model in `.keras` format         |
| OpenCV not installed | Installed via `pip install opencv-python` |
| Webcam not detected  | Used `cv2.VideoCapture(0, cv2.CAP_DSHOW)` |





