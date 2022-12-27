from mmcls.apis import init_model, inference_model
import mmcv
import time,os
import json
from clearml import Task
def validation_inference(validation_root_path,config_file,checkpoint_file):
    import glob
    import time
    model = init_model(config_file, checkpoint_file, device='cuda:1')
    model_name=os.path.basename(config_file).split(".")[0].split("_p")[0]

    validation_path_name=os.path.basename(validation_root_path)
    import platform
    from datetime import datetime as dt
    now = dt.now()
    timestr=now.strftime("%Y%m%d_%H:%M:%S")
    task = Task.init(project_name='CLS_VALIDATION', task_name=f"{model_name}-{validation_path_name}-{platform.node()}-{timestr}")
    Task.current_task().upload_artifact(name='config_file', artifact_object=config_file)
    # build the model from a config file and a checkpoint file
    
    image_list=glob.glob(f"{validation_root_path}/*/*.bmp")
    validation_output_path_root=f"{validation_root_path}_validation_result_output"
    counter=0
    bad_counter=0
    import shutil
    begin_time=time.time()
    
    inference_counter={}
    expected_counter={}
    discrepancy_counter={}
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
        
        inference_label=result2["pred_class"]
        if inference_label not in inference_counter:
            inference_counter[inference_label]=0
            print(f"Found inference label {inference_label}")


        expected_label=os.path.basename(os.path.dirname(image_file))
        if expected_label not in expected_counter:
            expected_counter[expected_label]=0
            output_path=f"{validation_output_path_root}/wrong_{expected_label}"
            os.makedirs(output_path,exist_ok=True)
            print(f"Found expected label {expected_label}")
        
        if expected_label not in discrepancy_counter:
            discrepancy_counter[expected_label]=0

        inference_counter[inference_label]=inference_counter[inference_label]+1
        expected_counter[expected_label]=expected_counter[expected_label]+1
        counter=counter+1
        
        if inference_label!=expected_label:
            discrepancy_counter[expected_label]=discrepancy_counter[expected_label]+1
            #mmcv.imwrite(inference_img,f"{output_path}/{result2['pred_label']}-{result2['pred_class']}-{result2['pred_score']}{model_name}_{filename}.jpg")
        if counter % 10 == 0:
            print(f"current counter {counter} discrepancy_counter {discrepancy_counter} expected_counter {expected_counter} inference_counter:{inference_counter}")
    

    result_stats={}
    stats={}
    for item in discrepancy_counter:
        if item not in stats:
            stats[item]=0
        
        stats[item]=f"{round(discrepancy_counter[item]/expected_counter[item]*100,2)}%"



    time_took_avg=f"{(time.time()-begin_time)/counter:.4f}"
    result_stats["discrepancy_counter"]=discrepancy_counter
    result_stats["inference_counter"]=inference_counter
    result_stats["expected_counter"]=expected_counter
    result_stats["incorrect_rate"]=stats

    result_stats["total_counter"]=counter
    result_stats["model_name"]=model_name
    result_stats["time_took_avg"]=time_took_avg
    result_stats["total_counter"]=counter
    result_stats["config_file"]=config_file
    result_stats["checkpoint_file"]=checkpoint_file
    result_stats["validation_root_path"]=validation_root_path
    print(f"{result_stats} ")
    Task.current_task().upload_artifact(name='result_stats', artifact_object=result_stats)
    json_object = json.dumps(result_stats, indent=4)
    from datetime import datetime
    
    # get current date and time
    current_datetime = datetime.now()
    print("Current date & time : ", current_datetime)
    
    # convert datetime obj to string
    str_current_datetime = str(current_datetime)
    
    # create a file object along with extension
    file_name = str_current_datetime+".json"
    # Writing to sample.json
    with open(f"{validation_output_path_root}/{file_name}", "w") as outfile:
        outfile.write(json_object)

config_file = '/opt/workspace/mmclassification/configs/joycv/resnet18_8xb16_package_cls_half_size_499_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/resnet18_8xb16_package_cls_half_size_499_1220/best_accuracy_top-1_epoch_61.pth'

config_file = '/opt/workspace/mmclassification/work_dirs/repvgg-D2-package_with_validation_cls_half_size_499_1220/repvgg-D2-package_with_validation_cls_half_size_499_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/repvgg-D2-package_with_validation_cls_half_size_499_1220/best_accuracy_top-1_epoch_13.pth'

config_file = '/opt/workspace/mmclassification/work_dirs/repvgg-A0-package_cls_half_size_499_1220/repvgg-A0-package_cls_half_size_499_1220.py'
checkpoint_file = '/opt/workspace/mmclassification/work_dirs/repvgg-A0-package_cls_half_size_499_1220/best_accuracy_top-1_epoch_17.pth'

validation_root = "/opt/workspace/imagedb/packs/trained/"
validation_root = "/opt/workspace/imagedb/packs/1202_untrained/"
validation_root = "/opt/workspace/imagedb/packs/untrained/"

validation_inference(validation_root,config_file,checkpoint_file)
  
