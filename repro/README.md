# Reproduce DDQ

# 1 Introduction

Server: [北京超级云计算中心](https://cloud.blsc.cn/)  



# 2 Dataset

Upload COCO 2017 data to `~/data/lvis` and run commands:

```
mkdir ~/ddq/legacy/proj/DDQ/data/
ln -s ~/data/lvis ~/ddq/legacy/proj/DDQ/data/coco
```

Then data under `~/ddq/legacy/proj/DDQ/data/coco` should be:

```
DDQ
├── data
│   ├── coco
│   │   ├── annotations
│   │   │      ├──instances_train2017.json
│   │   │      ├──instances_val2017.json
│   │   ├── train2017
│   │   ├── val2017
```



# 3 Train/Test Environment

Run bash commands:

```
mkdir -p ~/ddq/legacy/proj
cd ~/ddq/legacy/proj
git clone https://github.com/jiongjiongli/DDQ.git

python -m venv  ~/ddq/legacy/py38

cd ~/ddq/legacy/proj/DDQ
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/ddq/legacy/py38/bin/activate

pip install --upgrade pip
pip install cython numpy \
matplotlib pycocotools scipy shapely six terminaltables \
cityscapesscripts imagecorruptions scikit-learn \
pytest-runner \
ninja psutil \
addict numpy packaging Pillow pyyaml yapf coverage lmdb onnx onnxoptimizer onnxruntime pytest PyTurboJPEG scipy tifffile
# mmengine>=0.3.0

# Install pytorch.
# Option 1: install directly.
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Option 2: Download and then install.
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl

pip install ~/data/torch-1.9.0+cu111-cp38-cp38-linux_x86_64.whl
pip install ~/data/torchvision-0.10.0+cu111-cp38-cp38-linux_x86_64.whl

# Install mmcv.
cd ~/ddq/legacy/proj/DDQ/mmcv-1.4.7
MMCV_WITH_OPS=1 python setup.py build_ext --inplace

export PYTHONPATH=`pwd`:$PYTHONPATH

cd ~/ddq/legacy/proj/DDQ
```



# Train

```
cd ~/ddq/legacy/proj/DDQ
bash ./repro/train.sh
```



# Test



```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/ddq/legacy/py38/bin/activate

cd ~/ddq/legacy/proj/DDQ/mmcv-1.4.7
export PYTHONPATH=`pwd`:$PYTHONPATH

cd ~/ddq/legacy/proj/DDQ

python tools/train.py projects/configs/ddq_fcn/ddq_fcn_r50_1x.py --work-dir ./exp/ddq_fcn

python tools/test.py  projects/configs/ddq_fcn/ddq_fcn_r50_1x.py ~/data/pretrain_models/ddq_fcn_r50_1x.pth --eval bbox

python tools/test.py  projects/configs/ddq_fcn/ddq_fcn_r50_1x.py ~/data/pretrain_models/ddq_fcn_3x.pth --eval bbox

```

