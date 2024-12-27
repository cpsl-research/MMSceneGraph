#!/usr/bin/env bash

MODE=${1:-predcls}
MODEL=${2:-vtranse}

cd ..
SGG_CFG=configs/scene_graphs/${MODEL}/${MODEL}-${MODE}-faster-rcnn_r50_fpn_1x_visualgenome.py
python tools/train.py $SGG_CFG