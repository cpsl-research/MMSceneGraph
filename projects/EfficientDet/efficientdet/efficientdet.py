# Copyright (c) OpenMMLab. All rights reserved.
from mmscene.models.detectors.single_stage import SingleStageDetector
from mmscene.registry import MODELS
from mmscene.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class EfficientDet(SingleStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
