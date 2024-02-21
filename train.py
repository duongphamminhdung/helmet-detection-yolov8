from ultralytics import YOLO

# Load a model
model = YOLO("models/yolov8n.pt")
model.to('cuda')

# Using the model
model.train(data="data.yaml", epochs=) 