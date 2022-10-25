#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
UPPER_DIR=$SCRIPT_DIR/../
#echo $UPPER_DIR
cd $UPPER_DIR
echo "Working DIR: "$(pwd)

WORKING_DIR=$(pwd)/work_dir/efficientnet-b8norm
CONFIG_FILE="configs/joycv/efficientnet-b8norm-freshchustnut.py"

/opt/miniconda3/envs/mmclassification/bin/python tools/train.py  $CONFIG_FILE --work-dir=$WORKING_DIR

BEST_MODEL=$WORKING_DIR/$(ls $WORKING_DIR|grep -v 'torchscript'|grep -i 'best')
TARGET_TORCH_BEST_MODEL=$BEST_MODEL.torchscript.pt
#echo $BEST_MODEL

echo Source model : $BEST_MODEL
echo Target model : $TARGET_TORCH_BEST_MODEL

/opt/miniconda3/envs/mmclassification/bin/python tools/deployment/pytorch2torchscript.py  --checkpoint $BEST_MODEL  --output-file $TARGET_TORCH_BEST_MODEL   --verify   $CONFIG_FILE