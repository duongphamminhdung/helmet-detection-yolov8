#install requirements
echo pip install ultralytics
pip install ultralytics

#download models
cd ..
mkdir models
cd models
echo download yolov8 models: nano, small and medium
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt

#download data
cd ../data
curl -L "https://universe.roboflow.com/ds/jFHhzo5mX4?key=esYjvWKyyK" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

echo download data
