_base_ = 'mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'

# bs4: 7+3=10mins
# bs8: 6+3=9mins
# bs16: 5+3=8mins
# 每4h: 240mins
# 考虑bs8, 训练50, 每4h大概确保能够训练25个
# 考虑训练60+epoch

resume = 'auto'
max_epochs = 30
batch_size = 8
val_interval = 1
lr_base = 2e-5
mstone_first = 24
mstone_second = 28
# num of samples per gpu
num_stages = 6
num_proposals = 100
backend_args = None

dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'  

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


train_dataloader = dict(
    batch_size=batch_size,
    persistent_workers=True,  # 加速数据加载
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='ann_coco/voc12_train.json',
        data_prefix=dict(img=''),
        data_root=data_root,
        metainfo=dict(
        classes=classes
        )
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='ann_coco/voc12_val.json',
        data_prefix=dict(img=''),
        data_root=data_root,
        metainfo=dict(
        classes=classes
        )
    )
)

test_dataloader = dict(
    batch_size=batch_size,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='ann_coco/voc12_val.json',
        data_prefix=dict(img=''),
        data_root=data_root,
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
    type='SparseRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=20)
            for _ in range(num_stages)
        ],
    )
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,  
    val_interval=val_interval
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=lr_base, weight_decay=0.0001),
    clip_grad=dict(max_norm=1, norm_type=2)
)
# 学习率调度（缩短热身期）
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=800),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[mstone_first,mstone_second],  # 学习率衰减节点
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1))

# TensorBoard日志
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

work_dir = 'work_dirs/sparse_rcnn'


