# -----------------------------------------------------------------------------
# Code created by: Mohammed Safwanul Islam @safwandotcom®
# Project: Computer Vision Data Science OPENPOSE 
# Date created: 16th November 2024
# Organization: N/A
# -----------------------------------------------------------------------------
# Description:
# This code captures live video from the webcam, applies pose estimation using MediaPipe, 
# and visualizes the detected body landmarks and connections in real time. The program runs continuously until the user presses the 'q' key to exit. 
# It demonstrates an application of computer vision for human pose tracking, which can be used in fields like fitness, gaming, and gesture recognition.
#   VERSION 3 visualizes the data into a black pop-up window which opens after the program is successfully running.
#### This code can detect only single person in detection ####
# -----------------------------------------------------------------------------
# License:
# This code belongs to @safwandotcom®.
# Code can be freely used for any purpose with proper attribution.
# -----------------------------------------------------------------------------
# Modules to install for this program to run using WINDOWS POWERSHELL
# pip install opencv-python
# pip install mediapipe


import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam capture
cap = cv2.VideoCapture(1)  # Use 1 for external webcam, 0 for internal webcam

# List of connection colors (Blue, Light Orange, Cyan, Green, Red, Indigo, Violet)
CONNECTION_COLORS = [
    (0, 0, 255),     # Blue
    (0, 165, 255),   # Light Orange
    (0, 255, 255),   # Cyan
    (0, 255, 0),     # Green
    (255, 0, 0),     # Red
    (75, 0, 130),    # Indigo
    (238, 130, 238)  # Violet
]

# Function to generate a color gradient for landmarks
def generate_color_gradient(index, total):
    # Generate a color gradient from red to green to blue
    ratio = index / total
    r = int(255 * (1 - ratio))  # Red decreases
    g = int(255 * ratio)        # Green increases
    b = int(255 * (ratio))      # Blue increases
    return (b, g, r)  # Return in BGR format for OpenCV

# Function to generate a color for each connection based on the index
def generate_connection_color(index, total):
    # Use modulo to cycle through the predefined colors in the CONNECTION_COLORS list
    return CONNECTION_COLORS[index % len(CONNECTION_COLORS)]

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image from BGR (OpenCV default) to RGB (MediaPipe input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    pose_results = pose.process(image_rgb)

    # Create a black background to draw the pose landmarks and connections
    black_background = np.zeros(image.shape, dtype=np.uint8)


    # If pose landmarks are detected, draw them with gradient colors
    if pose_results.pose_landmarks:
        # Loop through all landmarks and apply a color gradient
        total_landmarks = len(pose_results.pose_landmarks.landmark)
        for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
            color = generate_color_gradient(i, total_landmarks)
            # Draw the individual landmark with the gradient color
            cv2.circle(black_background, 
                       (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 
                       5, color, -1)

        # Draw the pose connections with predefined colors
        for i, connection in enumerate(mp_pose.POSE_CONNECTIONS):
            start_idx, end_idx = connection
            start_landmark = pose_results.pose_landmarks.landmark[start_idx]
            end_landmark = pose_results.pose_landmarks.landmark[end_idx]

            # Get the color for the connection from predefined colors
            color = generate_connection_color(i, len(mp_pose.POSE_CONNECTIONS))
            # Draw the connection line with the selected color
            start_point = (int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0]))
            end_point = (int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0]))
            cv2.line(black_background, start_point, end_point, color, 2)

    # Display the black background with gradient pose landmarks and connections
    cv2.imshow('Open Pose Output by Safwanul', black_background)

    # Draw the landmarks and connections on the original image with gradients
    if pose_results.pose_landmarks:
        total_landmarks = len(pose_results.pose_landmarks.landmark)
        for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
            color = generate_color_gradient(i, total_landmarks)
            # Draw the individual landmark with the gradient color
            cv2.circle(image, 
                       (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 
                       5, color, -1)

        # Draw the pose connections with predefined colors
        for i, connection in enumerate(mp_pose.POSE_CONNECTIONS):
            start_idx, end_idx = connection
            start_landmark = pose_results.pose_landmarks.landmark[start_idx]
            end_landmark = pose_results.pose_landmarks.landmark[end_idx]

            # Get the color for the connection from predefined colors
            color = generate_connection_color(i, len(mp_pose.POSE_CONNECTIONS))
            # Draw the connection line with the selected color
            start_point = (int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0]))
            end_point = (int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0]))
            cv2.line(image, start_point, end_point, color, 2)

    # Display the original webcam feed with gradient pose landmarks and connections
    cv2.imshow('Webcam of Safwanul', image)

    # Exit on pressing the 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
