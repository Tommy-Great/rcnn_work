_base_ = 'mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_amp-1x_coco.py'


# bs8:
# bs16: 10+5=15mins
# 4h约处理20个;

# hyperparams:
resume = 'auto'
max_epochs = 30
batch_size = 8
val_interval = 1
lr_base = 2.5e-3
mstone_first = 24
mstone_second = 28
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'
backend_args = None

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        ann_file='ann_coco/voc12_train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        type=dataset_type,
        metainfo=dict(
            classes=classes
        )
    ),
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        ann_file='ann_coco/voc12_val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root=data_root,
        test_mode=True,
        type=dataset_type,
        metainfo=dict(
            classes=classes
        )
    )
)

test_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        ann_file='ann_coco/voc12_val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root=data_root,
        test_mode=True,
        type=dataset_type,
        metainfo=dict(
            classes=classes
        )
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'ann_coco/voc12_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            num_classes=20
        ),
        mask_head=dict(
            num_classes=20
        )
    )
)

# 学习率自动缩放
auto_scale_lr = dict(base_batch_size=16, enable=False)

# # 优化器配置 (启用混合精度训练)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=lr_base,
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=None
)

# learning rate
param_scheduler = [
    dict(
    type='LinearLR',  # 使用线性学习率预热
    start_factor=0.001, # 学习率预热的系数
    by_epoch=False,  # 按 iteration 更新预热学习率
    begin=0, 
    end=800), 
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[mstone_first,mstone_second],  # 在..个 epoch 降低学习率
        gamma=0.1)
]
# 训练和测试配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# 日志配置
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1))

# 可视化配置
vis_backends = [
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 工作目录设置
work_dir = 'work_dirs/mask_rcnn'






