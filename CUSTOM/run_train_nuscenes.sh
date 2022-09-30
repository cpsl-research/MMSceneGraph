python convert_nuscenes_labels.py
if [ $? -eq 0 ]; then
    echo 'CONVERSION OK -- RUNNING TRAINING'
    cd ..
	python tools/train.py configs/nuscenes/faster_rcnn_r50_fpn_1x_nuscenes.py
else
    echo 'CONVERSION FAILED -- EXITING'
fi

