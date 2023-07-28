# Reproduce DDQ

# 1 Introduction

## 1.1 任务拆分

| 阶段 | 是否需要训练模型 | 任务                                                         |
| ---- | ---------------- | ------------------------------------------------------------ |
| 1    | 否               | 使用开源DDQ+开源模型，测试，复现论文AP                       |
| 2    | 否               | 开发新版本DDQ推理功能，开源模型转权重以适配新版本，测试，与老版本结果对齐 |
| 3    | 是               | 使用开源DDQ，训练模型，测试，复现DDQ结果                     |
| 4    | 是               | 开发新版本DDQ训练功能，训练模型，测试，与老版本结果对齐      |



## 1.2 需要支持的模型和训练配置

| 是否需要训练 | Model           | Backbone | Lr schd | Augmentation |
| ------------ | --------------- | -------- | ------- | ------------ |
| 否           | DDQ FCN         | R-50     | 12e     | Normal       |
| 否           | DDQ FCN         | R-50     | 36e     | DETR         |
| 否           | DDQ R-CNN       | R-50     | 12e     | Normal       |
| 否           | DDQ R-CNN       | R-50     | 36e     | DETR         |
| 是           | DDQ DETR-4scale | R-50     | 12e     | DETR         |
| 否           | DDQ DETR-5scale | R-50     | 12e     | DETR         |
| 否           | DDQ DETR-4scale | Swin-L   | 30e     | DETR         |



- 要check in 的分支：[MMDetection dev-3.x](https://github.com/open-mmlab/mmdetection/tree/dev-3.x)
- 训练要使用论文中的配置，如使用8个GPU，每个服务器2个sample。
- 服务器: [北京超级云计算中心](https://cloud.blsc.cn/)  




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

## 3.1 Legacy Envonronment

### 3.1.1 DDQ FCN&R-CNN

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
addict numpy packaging Pillow pyyaml yapf coverage lmdb onnx onnxoptimizer onnxruntime pytest PyTurboJPEG scipy tifffile \
ipython
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



### 3.1.2 DDQ DETR



## 3.2 New Environment





# 4 Train

## Submit training job

```
cd ~/ddq/legacy/proj/DDQ
chmod +x ./repro/train.sh
dsub -s ./repro/train.sh
```



```
cd ~/ddq/legacy/proj/DDQ
chmod +x ./repro/gpu_4_train.sh
dsub -s ./repro/gpu_4_train.sh
```



## Show / Cancel jobs

```
# Show running job id and status
djob

# Cancel job
djob -T job_id
```



# 5 Test



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





# Test

## test.main

```
tools/test.py

def main():
    ...
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    cfg.data.test.test_mode = True
    ...
    # type = 'CocoDataset'
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False)
    
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, ...)
        
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, ...)
```



## build_dataset

```
mmdet/datasets/builder.py

DATASETS = Registry('dataset')

def build_dataset(cfg, default_args=None):
    # -> mmcv.utils.build_from_cfg
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    
    return dataset
```



## build_detector

```
mmdet\models\builder.py

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
DETECTORS = MODELS

def build_detector(cfg, train_cfg=None, test_cfg=None):
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
```



## mmcv.cnn.MODELS

```
mmcv-1.4.7\mmcv\cnn\builder.py

MODELS = Registry('model', build_func=build_model_from_cfg)
```



## build_from_cfg

```
mmcv-1.4.7\mmcv\utils\registry.py

def build_from_cfg(cfg, registry, default_args=None):
    args = cfg.copy()
    obj_type = args.pop('type')
    
    obj_cls = registry.get(obj_type)
    
    return obj_cls(**args)
```





## CocoDataset

### CocoDataset.load_annotations

```
mmdet\datasets\coco.py

class CocoDataset(CustomDataset):
        def load_annotations(self, ann_file):
            self.coco = COCO(ann_file)
            
```



## CustomDataset

### CustomDataset.init

```
mmdet\datasets\custom.py

class CustomDataset(Dataset):

    def __init__(...):
        # Inputs:
            ann_file='data/coco/annotations/instances_val2017.json',
            img_prefix='data/coco/val2017/',
            pipeline=test_pipeline,
            test_mode = True
        #
        
        # CocoDataset.CLASSES
        self.CLASSES = self.get_classes(classes)
        
        # self.file_client = FileClient inst, inst.client=HardDiskBackend()
        self.file_client = mmcv.FileClient(**file_client_args)
        
        # local_path=self.ann_file
        # self.data_infos = [image_info for each image]
        #     image_info['filename'] = image_info['file_name']
        # -> CocoDataset.load_annotations(local_path)
        self.data_infos = self.load_annotations(local_path)
        
        self.pipeline = Compose(pipeline)
```



### CustomDataset.getitem

```
class CustomDataset(Dataset):
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
```



### CustomDataset.prepare_test_img

```
class CustomDataset(Dataset):
    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        # {'img_info': img_info}
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
            
        # 
            {
                'img_info':      img_info,
                'img_prefix':    'data/coco/val2017/',
                'seg_prefix':    None,
                'proposal_file': None,
                'bbox_fields':   [],
                'mask_fields':   [],
                'seg_fields':    []
            }
        self.pre_pipeline(results)
        
        return self.pipeline(results)
```



## CocoDataset

### CocoDataset.load_annotations

```
mmdet\datasets\coco.py

class CocoDataset(CustomDataset):
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        
        # [category_id for each category in database['categories'] if category in self.CLASSES]
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        
        # {category_id: category_index for each category in database['categories'] 
        #    if category in self.CLASSES}
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        # [image_id for each image]
        self.img_ids = self.coco.get_img_ids()
        
        # [image_info for each image]
        #     image_info['filename'] = image_info['file_name']
        return data_infos
```



## single_gpu_test

## multi_gpu_test

```
mmdet\apis\test.py

def single_gpu_test(...):
or
def multi_gpu_test(...):
    model.eval()

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
```





# COCO

## COCO.init

```
cocoapi\PythonAPI\pycocotools\coco.py

class COCO:
    def __init__(self, annotation_file=None):
        # json content of annotation_file
        self.dataset
        self.createIndex()
```



## COCO.createIndex

```
    def createIndex(self):
        # {annotation_id: annotation_info}
        self.anns
        
        # {image_id: [annotation_info in the image]}
        self.imgToAnns
        
        # {image_id: image_info}
        self.imgs
        
        # {category_id: category}
        self.cats
        
        # {category_id: [image_id in the category]}
        self.catToImgs
```

