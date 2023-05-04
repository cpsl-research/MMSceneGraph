# The new config inherits a base config to highlight the necessary modification
_base_ = ['../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))

# Modify dataset related settings
dataset_type = 'nuScenesDataset'
classes = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
data = dict(
    train=dict(
        # img_prefix='data/nuscenes/data',
        img_prefix = '',
        classes=classes,
        ann_file='data/nuscenes/train_annotation_nuscenes_in_coco.json'),
    val=dict(
        img_prefix='data/nuscenes/data',
        classes=classes,
        ann_file='data/nuscenes/val_annotation_nuscenes_in_coco.json'),
    test=dict(
        # img_prefix='data/nucenes/data',
        img_prefix = '',
        classes=classes,
        ann_file='data/nuscenes/val_annotation_nuscenes_in_coco.json'))

# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=8)
log_config = dict(interval=100)

# We can use the pre-trained model to obtain higher performance
load_from = 'checkpoints/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth'
