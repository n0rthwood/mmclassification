_base_ = [
    '../_base_/models/repvgg-A0_in1k.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

train_path='/opt/workspace/imagedb/package_cls_data/DatasetId_1725538_1669651240'
test_path='/opt/workspace/imagedb/package_cls_data/DatasetId_1725538_1669651240'
train_max_epochs=300
#load_from =  "work_dir/run2/latest.pth"

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
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
#classes = train_classes

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

img_test_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=(999,2440)),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=(999,2440)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    
    samples_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix=train_path_images,
        ann_file=train_path_annotation,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix=train_path_images,
        ann_file=train_path_annotation,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix=train_path_images,
        ann_file=train_path_annotation,
        pipeline=test_pipeline
    )
)