#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import face_recognition
import os
import numpy as np

def load_known_faces_from_directory(base_path):
    """
    Load face encodings from images in directories of known faces.

    Args:
    base_path (str): Directory where known faces are stored.

    Returns:
    dict: A dictionary with person names as keys and lists of face encodings as values.
    """
    known_faces = {}
        person_dir = os.path.join(base_path, person_name)
        if os.path.isdir(person_dir):
            known_faces[person_name] = []
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_faces[person_name].extend(encodings)
    return known_faces

def main():
    base_path = r'known_faces'  # Directory where known faces are stored
    known_faces = load_known_faces_from_directory(base_path)
    
    known_face_encodings = []
    known_face_names = []

    # Flatten the known_faces dictionary
    for name, encodings in known_faces.items():
        known_face_names.extend([name] * len(encodings))
        known_face_encodings.extend(encodings)
    
    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Convert the frame to RGB (face_recognition works with RGB images)
        rgb_frame = frame[:, :, ::-1]
        
        # Detect faces and get face encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the detected face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # Check if the best match is below the distance threshold
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
            
            # Draw a rectangle around the face and add the name tag
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Exit the video feed on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




