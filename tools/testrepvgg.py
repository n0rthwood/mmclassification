from mmcls.apis import init_model, inference_model
import mmcv
import time,os
# Specify the path to model config and checkpoint file
config_file = '/opt/workspace/mmclassification/configs/joycv/repvgg-A0-package_cls_half_size_500_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/repvgg-A0-package_cls_half_size_500_1220/backup/good-0d25-bad-2.pth'
#config_file = '/opt/workspace/mmclassification/configs/joycv/repvgg-D2-package_with_validation_cls_half_size_499_1220.py'
#checkpoint_file = '/opt/workspace/mmclassification/work_dirs/repvgg-D2-package_with_validation_cls_half_size_499_1220/best_accuracy_top-1_epoch_13.pth'


# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

#img_path="/opt/workspace/imagedb/package_cls_data/DatasetId_1725538_1669651240/Images/"
ok_img_path="/opt/workspace/imagedb/packs/1211/"
#bad_img_path="/opt/workspace/imagedb/package_cls_data/DatasetId_1733993_1670733467/sp2"



import shutil
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
        #print(image_file)
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
        
        if result2["pred_label"]==2 and result2["pred_score"]>0.7:
            bad_counter=bad_counter+1
            print(filename)
            #print(result2["pred_score"])
            mmcv.imwrite(inference_img,f"{output_path}/{result2['pred_label']}-{result2['pred_class']}-{result2['pred_score']}_{filename}.jpg")
            #mmcv.imwrite(inference_img,f"{output_path}/{filename}.jpg")
            #shutil.copy(image_file, f"{output_path}/{filename}.bmp")
    print(f"total image{counter},bad {bad_counter} percent{100*bad_counter/counter:.2f} avg timetook:{(time.time()-begin_time)/counter:.4f}")
inference(ok_img_path)
#inference(bad_img_path)    
