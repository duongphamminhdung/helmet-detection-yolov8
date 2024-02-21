python train.py \
        --path "/root/helmet-detection-yolov8/models/yolov8s.pt"\
        --cuda \
        --epochs 100 \
        --batch_size 32 \
        --num_workers 8 \
        --seed 2506 \
        --name "21Feb" \
        --save_period 3 \
        # --save_folder "" \
