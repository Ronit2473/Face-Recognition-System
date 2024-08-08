# Face-Recognition-System
The Face Recognition System is a Python-based application designed to identify and recognize faces in real-time using a webcam. The system utilizes the face_recognition library for detecting and encoding faces and the cv2 library (OpenCV) for capturing video feed and drawing overlays.

The application loads known face encodings from a specified directory, where each subdirectory represents a different individual. During real-time operation, the system captures frames from the webcam, detects faces, and compares them against the stored encodings to identify known individuals. If a match is found, the person's name is displayed on the screen along with a bounding box around their face.

This project can be used for security systems, attendance tracking, or any application that requires facial recognition.
