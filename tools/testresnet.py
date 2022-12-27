from mmcls.apis import init_model, inference_model
import mmcv
import time,os
# Specify the path to model config and checkpoint file
config_file = '/opt/workspace/mmclassification/configs/joycv/resnet18_8xb16_package_cls_half_size_499_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/resnet18_8xb16_package_cls_half_size_499_1220/best_accuracy_top-1_epoch_61.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:1')
model_name=os.path.basename(config_file).split(".")[0].split("_")[0:1]
#img_path="/opt/workspace/imagedb/package_cls_data/DatasetId_1725538_1669651240/Images/"
ok_img_path="/opt/workspace/imagedb/packs/testset_good"
bad_img_path="/opt/workspace/imagedb/packs/testset2_used"


def inference(image_path):
    import glob
    import time
    image_list=glob.glob(f"{image_path}/*.bmp")
    output_path=f"{image_path}/out/"
    os.makedirs(output_path,exist_ok=True)
    counter=0
    bad_counter=0
    import shutil
    begin_time=time.time()
    for image_file in image_list:
        img = mmcv.imread(image_file)
        #img = mmcv.imrescale(img,scale=(500,1220,3))     
        #print(image_file)
        filename=os.path.basename(image_file)
        filename=filename.split(".")[0]
        image_begin_time=time.time()
        inference_img=img
        result2 = inference_model(model,inference_img )
        image_end_time=time.time()
        #print(f"single image timetook:{image_end_time-image_begin_time:.4f}")
        counter=counter+1
        if result2["pred_class"]=="bad":
            bad_counter=bad_counter+1
            mmcv.imwrite(inference_img,f"{output_path}/{result2['pred_label']}-{result2['pred_class']}-{result2['pred_score']}{model_name}_{filename}.jpg")
    print(f"{model_name} total image{counter},bad {bad_counter} percent{100*bad_counter/counter:.2f} avg timetook:{(time.time()-begin_time)/counter:.4f}")
inference(ok_img_path)
inference(bad_img_path)    
