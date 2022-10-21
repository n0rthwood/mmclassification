import os.path


import numpy as np
import json
if __name__ == '__main__':
    import sys
    sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/../")
    for item in sys.path:
        print(item)
    from mmcls.datasets.builder import DATASETS
    from mmcls.datasets.base_dataset import BaseDataset
else:
    from .builder import DATASETS
    from .base_dataset import BaseDataset







@DATASETS.register_module()
class BaiduCocoJsonList(BaseDataset):


    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        annotation_file_obj = open(self.ann_file)
        annotation_data_obj = json.load(annotation_file_obj)
        images_info = annotation_data_obj['images']
        annotations_info = annotation_data_obj['annotations']
        categories_info = annotation_data_obj['categories']
        category_info_sorted = sorted(categories_info, key=lambda d: d['id'])

        self.CLASSES=[]
        for cat_item in category_info_sorted:
            self.CLASSES.append(cat_item['name'])


        data_infos = []

        def __find_filename(images_info, image_id):
            for image_item in images_info:
                if image_item['id'] == image_id:
                    return image_item['file_name']
            return None

        for annotation_item in annotations_info:
            category_id=annotation_item['category_id']
            image_id=annotation_item['image_id']
            filename=__find_filename(images_info,image_id)
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(category_id, dtype=np.int64)
            data_infos.append(info)
        return data_infos


if __name__ == '__main__':
    dataset = BaiduCocoJsonList(
        ann_file='/opt/images/fresh_chestnut/dataset_coco_json/Annotations/coco_info.json',
        data_prefix='/opt/images/fresh_chestnut/dataset_coco_json/Images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(224, 224), backend='pillow'),
            dict(type='Collect', keys=['img', 'gt_label'])
        ])
    print(dataset.CLASSES)
    print(len(dataset.CLASSES))
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])

