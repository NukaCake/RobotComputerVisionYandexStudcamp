# YandexUrfuStudcampCV
Computer vision and camera solutions for robot navigation on Yandex&URFU Robotics and AI Studcamp

Calibrate.py - script used to find optimal parameters for a cam to remove distortion(fisheye effect)
camera_write.py - cap tures videos from some camera
camera_yolo_show_from_file.py - run yolo on a video for testing
convert_map.py - creating grid on a map of robot arena and implementing a* algorithm
detection.py - isn't used, just some piece of chinese code from a box
home_camera.py - contains class TopCamera that calibrates, undistorts and crops image from above
live.py - using YOLO11 trained on a custom dataset on a top camera
test_yolo.py - for testing YOLO when webcam is used
undistorter.py - old version of undistorter
video_splitter_main.py - script used to split videos to frames and deleting similar ones depending on their embeddings

