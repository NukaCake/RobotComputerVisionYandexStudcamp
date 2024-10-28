import cv2
from ultralytics import YOLO
import time
import time
import numpy as np
from home_camera import *


input_video_path = 'yolov11/output_video3.avi' #path to video to detect images with yolo
camera = TopCamera()

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
model = YOLO('new.pt')
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()
    """frame = camera.undistort(frame)
    frame = camera.warp_transform(frame)
    try:
        frame = camera.crop_in_the_middle(frame)
        cv2.imwrite('left/cropped/' + file, frame)
    except:
        print('ERROR', file)"""
    annotated_frame = camera.undistort(annotated_frame)
    annotated_frame = camera.warp_transform(annotated_frame)
    annotated_frame = camera.crop_in_the_middle(annotated_frame)

    # Display the frame
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()