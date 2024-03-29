#config git
# git config --global user.email "duongphamminhdung@gmail.com"
# git config --global user.name "duongphamminhdung"

#install requirements
echo pip install ultralytics
pip install ultralytics imgaug

#download models
cd ../models
echo download yolov8 models: nano, small and medium
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -O ../models/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt -O ../models/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt -O ../models/yolov8m.pt

#download data
cd ../data
curl -L "https://app.roboflow.com/ds/7sVYTMoSiw?key=MCtAccwOGv" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip  #custom dataset with augmentation
# curl -L "https://universe.roboflow.com/ds/jFHhzo5mX4?key=esYjvWKyyK" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
echo download data

#install nvtop and htop
apt install nvtop htop

tmux new -s 2506
