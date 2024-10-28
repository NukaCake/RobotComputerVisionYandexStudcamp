import cv2
from ultralytics import YOLO
import time
from home_camera import *


#Connect to the up camera
cap = cv2.VideoCapture('rtsp://Admin:rtf123@192.168.2.250/251:554/1/1')

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
#model for detecting objects from above camera
model = YOLO('best_up.pt')
i = 0
camera = TopCamera()
while True:
    ret, frame = cap.read()
    if i % 5 == 0:
        ret, frame = cap.read()
        if not ret:
            print('End')
            break
        frame = camera.undistort(frame)
        frame = camera.warp_transform(frame)
        frame = camera.crop_in_the_middle(frame)
        #frame = camera.binarize(frame)

        # Perform object detection
        results = model(frame)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow('YOLO Object Detection', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()