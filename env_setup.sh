conda env create --name mmclassification python=3.8
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
eval "$(conda shell.bash hook)" && conda activate mmclassification
python3 -m pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
pip install future tensorboard
mim install mmcv-full
pip install onnxruntime  -i https://pypi.douban.com/simple/
pip install -v -e .
pip install -r $SCRIPT_DIR/requirements.txt