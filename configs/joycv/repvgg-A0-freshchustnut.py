_base_ = [
    '../_base_/models/repvgg-A0_in1k.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=10)
#schedules override
runner = dict(type='EpochBasedRunner', max_epochs=500)
# dataset settings
dataset_type = 'BaiduCocoJsonList'
classes = ('wormwhole', 'mod', 'good', 'empty', 'crack')
#
# model settings
model = dict(
    head=dict(
        num_classes=len(classes),
    ))

img_norm_cfg = dict(
     mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5],to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=224,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    train=dict(
        type=dataset_type,
        classes=classes,
        data_prefix='/opt/images/fresh_chestnut/dataset_coco_json/Images/',
        ann_file='/opt/images/fresh_chestnut/dataset_coco_json/Annotations/coco_info.json',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_prefix='/opt/images/fresh_chestnut/dataset_coco_json/Images/',
        ann_file='/opt/images/fresh_chestnut/dataset_coco_json/Annotations/coco_info.json',
        pipeline=train_pipeline
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        data_prefix='/opt/images/fresh_chestnut/dataset_coco_json/Images/',
        ann_file='/opt/images/fresh_chestnut/dataset_coco_json/Annotations/coco_info.json',
        pipeline=test_pipeline
    )
)