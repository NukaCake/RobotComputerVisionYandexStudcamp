from ultralytics import YOLO
model = YOLO('new.pt')
#testing yolo model from a local webcam
results = model(0, show=True)

for result in results:
    boxes = result.boxes
    classes = result.names
