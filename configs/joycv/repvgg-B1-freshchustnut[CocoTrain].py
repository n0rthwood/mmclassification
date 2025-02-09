_base_ = '../repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py'

model = dict(backbone=dict(arch='B1'), head=dict(in_channels=2048))

train_max_epochs=1000

train_path='/opt/imagedb/train'
test_path='/opt/imagedb/test'


#load_from =  "/opt/workspace/mmclassification/work_dir/repvgg_b1_run1/best_accuracy_top-1_epoch_380.pth"
#resume_from = "/opt/workspace/mmclassification/work_dir/repvgg_b1_run1/latest.pth"
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])


train_path_annotation=f"{train_path}/Annotations/coco_info.json"
train_path_images=f"{train_path}/Images/"


def extract_category_info(annotation_path):
    import json
    annotation_info=None
    annotation_info=json.load(open(annotation_path))
    result = []
    categories_info = annotation_info['categories']
    category_info_sorted = sorted(categories_info, key=lambda d: d['id'])
    for cat_item in category_info_sorted:
            result.append(cat_item['name'])
    if len(result) ==0:
        raise Exception(f"Annotation classes is empty. check annotation file {annotation_path}")
    return result

train_classes=extract_category_info(train_path_annotation)

checkpoint_config = dict(interval=50)
evaluation = dict(interval=1, start=1, metric='accuracy', metric_options={'topk': (1, )}, save_best='accuracy_top-1')

# model settings
model = dict(
    head=dict(
        num_classes=len(train_classes),
    ))

#schedules override
runner = dict(type='EpochBasedRunner', max_epochs=train_max_epochs)
# dataset settings
dataset_type = 'BaiduCocoJsonList'
classes = train_classes

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

img_test_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
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
        data_prefix=train_path_images,
        ann_file=train_path_annotation,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_prefix=train_path_images,
        ann_file=train_path_annotation,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        data_prefix=f"{test_path}/Images/",
        ann_file=f"{test_path}/Annotations/coco_info.json",
        pipeline=test_pipeline
    )
)