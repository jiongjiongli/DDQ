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
# DDQ FCN&R-CNN
mkdir ~/ddq/legacy/proj/DDQ/data/
ln -s ~/data/lvis ~/ddq/legacy/proj/DDQ/data/coco


# DDQ DETR
mkdir ~/ddq/legacy/proj/ddq_detr/data/
ln -s ~/data/lvis ~/ddq/legacy/proj/ddq_detr/data/coco


# New DDQ DETR
mkdir ~/ddq/new/mmdetection/data/
ln -s ~/data/lvis ~/ddq/new/mmdetection/data/coco
```

Then data under `~/ddq/legacy/proj/DDQ/data/coco` or `~/ddq/legacy/proj/ddq_detr/data/coco` should be:

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

cd ~/ddq/legacy/proj/DDQ
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

python -m venv  ~/ddq/legacy/py38
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

```
cd ~/ddq/legacy/proj
git clone --branch ddq_detr https://gitee.com/jiongjiongli/DDQ.git ddq_detr

cd ~/ddq/legacy/proj/ddq_detr
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

python -m venv  ~/ddq/legacy/detr_py38
source ~/ddq/legacy/detr_py38/bin/activate

pip install --upgrade pip
pip install cython numpy \
matplotlib pycocotools scipy shapely six terminaltables \
cityscapesscripts imagecorruptions scikit-learn \
pytest-runner \
ninja psutil \
addict numpy packaging Pillow pyyaml yapf coverage lmdb onnx onnxoptimizer onnxruntime pytest PyTurboJPEG scipy tifffile \
termcolor rich \
dadaptation lion-pytorch parameterized \
ipython
# mmengine==0.6.0

# Install pytorch.
# Option 1: install directly.
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Option 2: Download and then install.
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl

pip install ~/data/torch-1.9.0+cu111-cp38-cp38-linux_x86_64.whl
pip install ~/data/torchvision-0.10.0+cu111-cp38-cp38-linux_x86_64.whl

# Install mmcv.
cd mmcv-2.0.0rc4
MMCV_WITH_OPS=1 python setup.py build_ext --inplace
# ln -s mmcv-2.0.0rc4/mmcv ./
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ~/ddq/legacy/proj/ddq_detr
```



## 3.2 New Environment

### 3.2.1 DDQ DETR

```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

mkdir ~/ddq/new
cd ~/ddq/new/

git clone -b jiongjiongli/ddq_detr https://gitee.com/jiongjiongli/mmdetection_dev.git mmdetection

cd ~/ddq/new/mmdetection/
python -m venv  ~/ddq/new/detr_py38
source ~/ddq/new/detr_py38/bin/activate

pip install --upgrade pip
pip install cython numpy \
matplotlib pycocotools scipy shapely six terminaltables \
cityscapesscripts imagecorruptions scikit-learn \
pytest-runner \
ninja psutil \
addict numpy packaging Pillow pyyaml yapf coverage lmdb onnx onnxoptimizer onnxruntime pytest PyTurboJPEG scipy tifffile \
termcolor rich \
dadaptation lion-pytorch parameterized \
ipython

# Install pytorch.
# Option 1: install directly.
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Option 2: Download and then install.
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl

pip install ~/data/torch-1.9.0+cu111-cp38-cp38-linux_x86_64.whl
pip install ~/data/torchvision-0.10.0+cu111-cp38-cp38-linux_x86_64.whl

pip install mmengine==0.7.1
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html

cd ~/ddq/new/mmdetection/
pip install -v -e .

# Test
pip install asynctest \
cityscapesscripts \
codecov \
flake8 \
imagecorruptions \
instaboostfast \
interrogate \
isort==4.3.21
# Note: used for kwarray.group_items, this may be ported to mmcv in the future. 
kwarray \
memory_profiler \
# -e git+https://github.com/open-mmlab/mmtracking@dev-1.x#egg=mmtrack \

pip install -e git+https://gitee.com/jiongjiongli/mmtracking@dev-1.x#egg=mmtrack
pip install nltk \
onnx==1.7.0 \
onnxruntime>=1.8.0 \
parameterized \
prettytable \
protobuf<=3.20.1 \
psutil \
pip install pytest \
transformers \
ubelt \
xdoctest>=0.10.0 \
yapf

```



### 3.2.2 DDQ DETR Colab 

```
!pip install --upgrade pip
!pip install cython numpy \
matplotlib pycocotools scipy shapely six terminaltables \
cityscapesscripts imagecorruptions scikit-learn \
pytest-runner \
ninja psutil \
addict numpy packaging Pillow pyyaml yapf coverage lmdb onnx onnxoptimizer onnxruntime pytest PyTurboJPEG scipy tifffile \
termcolor rich \
dadaptation lion-pytorch parameterized \
ipython

!pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

cd /content
!git clone -b jiongjiongli/ddq_detr https://gitee.com/jiongjiongli/mmdetection_dev.git mmdetection

cd /content/mmdetection/
!pip install -v -e .


```





# 4 Train

## 4.1 Submit training job

### 4.1.1 DDQ FCN&R-CNN

```
cd ~/ddq/legacy/proj/DDQ
cp ./repro/train.sh ./repro/launch_train.sh
chmod +x ./repro/launch_train.sh
dsub -s ./repro/launch_train.sh
```



```
cd ~/ddq/legacy/proj/DDQ
chmod +x ./repro/gpu_4_train.sh
dsub -s ./repro/gpu_4_train.sh
```



### 4.1.2 DDQ DETR

#### 4.1.2.1 Multi GPU

```
cd ~/ddq/legacy/proj/ddq_detr
cp ./repro/train.sh ./repro/launch_train.sh
chmod +x ./repro/launch_train.sh
dsub -s ./repro/launch_train.sh
```



#### 4.1.2.2 Single GPU

```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/ddq/legacy/detr_py38/bin/activate

cd mmcv-2.0.0rc4
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ~/ddq/legacy/proj/ddq_detr

python tools/train.py projects/configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py --work-dir ./exp/ddq_detr

python tools/train.py projects/configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py --work-dir ./exp/ddq_detr_seed7 --cfg-options randomness.seed=7

# rm -rf ./exp/ddq_detr_seed7_det

python tools/train.py projects/configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py --work-dir ./exp/ddq_detr_seed7_det --cfg-options randomness.seed=7 randomness.deterministic=True

```



### 4.1.3 New DDQ DETR

#### 4.1.3.1 Single GPU

```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/ddq/new/detr_py38/bin/activate

cd ~/ddq/new/mmdetection/

python tools/train.py configs/ddq//ddq-detr-4scale_r50_8xb2-12e_coco.py --work-dir ./exp/ddq_detr

python tools/train.py configs/ddq//ddq-detr-4scale_r50_8xb2-12e_coco.py --work-dir ./exp/ddq_detr_seed7 --cfg-options randomness.seed=7

# rm -rf ./exp/ddq_detr_seed7_det

python tools/train.py configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py --work-dir ./exp/ddq_detr_seed7_det --cfg-options randomness.seed=7 randomness.deterministic=True

```





```
export CUDA_VISIBLE_DEVICES=-1

cp configs/ddq//ddq-detr-5scale_r50_8xb2-12e_coco.py configs/ddq//train_ddq-detr-5scale_r50_8xb2-12e_coco.py

vi configs/ddq//train_ddq-detr-5scale_r50_8xb2-12e_coco.py

# default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=False))

python tools/train.py configs/ddq//train_ddq-detr-5scale_r50_8xb2-12e_coco.py --work-dir ./exp/ddq_detr

cp configs/ddq//ddq-detr-4scale_swinl_8xb2-30e_coco.py configs/ddq//train_ddq-detr-4scale_swinl_8xb2-30e_coco.py

vi configs/ddq//train_ddq-detr-4scale_swinl_8xb2-30e_coco.py
# init_cfg=None
# default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=False))

python tools/train.py configs/ddq//train_ddq-detr-4scale_swinl_8xb2-30e_coco.py --work-dir ./exp/ddq_detr

```



```
pytest tests/test_models/test_dense_heads/test_ddq_detr_head.py 

pytest tests/test_models/test_detectors/test_ddq_detr.py 

pytest tests/test_models/test_layers/test_transformer.py

pytest tests/test_models/test_losses/test_loss.py

pytest tests/test_models/test_task_modules/test_assigners/test_topk_hungarian_assigner.py

```







## 4.2 Show / Cancel jobs

```
# Show running job id and status
djob

# Cancel job
djob -T job_id
```



# 5 Test

## 5.1 DDQ FCN&R-CNN

```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/ddq/legacy/py38/bin/activate

cd ~/ddq/legacy/proj/DDQ/mmcv-1.4.7
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ~/ddq/legacy/proj/DDQ

python tools/train.py projects/configs/ddq_fcn/ddq_fcn_r50_1x.py --work-dir ./exp/ddq_fcn

python tools/test.py projects/configs/ddq_fcn/ddq_fcn_r50_1x.py ~/data/pretrain_models/ddq_fcn_r50_1x.pth --eval bbox

python tools/test.py projects/configs/ddq_fcn/ddq_fcn_r50_1x.py ~/data/pretrain_models/ddq_fcn_3x.pth --eval bbox


python tools/test.py projects/configs/ddq_fcn/ddq_fcn_r50_1x.py exp/ddq_fcn_4gpu/epoch_12.pth --eval bbox
```



## 5.2 DDQ DETR

```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/ddq/legacy/detr_py38/bin/activate

cd ~/ddq/legacy/proj/ddq_detr/mmcv-2.0.0rc4
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ~/ddq/legacy/proj/ddq_detr

python tools/test.py projects/configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py ~/data/pretrain_models/ddq_detr_4scale_coco_1x.pth

python tools/test.py projects/configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py exp/ddq_detr/epoch_12.pth



```





```
PORT=50136 sh tools/dist_test.sh configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py ~/data/pretrain_models/ddq_detr_4scale_coco_1x.pth 8 --eval bbox

python tools/test.py configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py ~/data/pretrain_models/ddq_detr_4scale_coco_1x.pth



```





```

```



# Source Code

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





# DDQ DETR.Test

## test.main

```
tools/test.py

def main():
    # Config instance.
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    
    # args.work_dir or ./work_dirs / config file stem
    cfg.work_dir
    cfg.load_from = args.checkpoint
    
    # runner: Runner instance.
     -> 
         ## DDQDETR instance.
         self.model = self.build_model(model)
        self.model = self.wrap_model(self.cfg.get('model_wrapper_cfg'), self.model)
    #
    runner = RUNNERS.build(cfg)
    
    runner.test()
     ->    
        self._test_loop = self.build_test_loop(self._test_loop)
        -> loop = LOOPS.build(
                    loop_cfg,
                    default_args=dict(
                        runner=self,
                        dataloader=self._test_dataloader,
                        evaluator=self._test_evaluator))
            ->
                TestLoop.__init__(...)
                    -> BaseLoop.__init__(...)
                        -> self.dataloader = runner.build_dataloader(dataloader, 
                                                                     seed=runner.seed, 
                                                                     diff_rank_seed=diff_rank_seed)
                           -> 
                                   dataset = DATASETS.build(dataset_cfg)
                                   ->     CocoDataset: BaseDetDataset: BaseDataset __init__
                                       self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
                                       self._join_prefix()
                                       
                                       self.pipeline = Compose(pipeline)
                                       -> pipeline.transforms = [TRANSFORMS.build(transform) 
                                                                 for each transform]
                                       
                                       self.full_init()
                                       -> 
                                        # Generate data info for each image
                                        {
                                            'img_path': self.data_prefix['img'] / img_info['file_name']
                                            'img_id': img_info['id'],
                                            'seg_map_path': None,
                                            'height': img_info['height'],
                                            'width': img_info['width'],
                                            'instances': [
                                                            {
                                                                'ignore_flag': 1 if iscrowd else 0,
                                                                'bbox': [x1, y1, x2, y2],
                                                                'bbox_label': category index,
                                                                # If segmentation exists:
                                                                    'mask': annotation['segmentation'], 
                                                                    # shape: [1, 2 * num_points]
                                                            }
                                                         ]
                                        }
                                   
                                   dataset.full_init()
                                   
                                   sampler = DATA_SAMPLERS.build(
                                    sampler_cfg,
                                    default_args=dict(dataset=dataset, seed=sampler_seed))
                                    
                                batch_sampler = DATA_SAMPLERS.build(
                                    batch_sampler_cfg,
                                    default_args=dict(
                                        sampler=sampler,
                                        batch_size=dataloader_cfg.pop('batch_size')))
                                        
                                collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                            dict(type='pseudo_collate'))
                                collate_fn_type = collate_fn_cfg.pop('type')
                                collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)
                                collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
                                
                                data_loader = DataLoader(
                                    dataset=dataset,
                                    sampler=sampler if batch_sampler is None else None,
                                    batch_sampler=batch_sampler,
                                    collate_fn=collate_fn,
                                    worker_init_fn=init_fn,
                                    **dataloader_cfg)
                    
                    self.evaluator = runner.build_evaluator(evaluator)
        self.load_or_resume()
        
        metrics = self.test_loop.run()
         -> TestLoop.run():
             self.runner.model.eval()
             
             for idx, data_batch in enumerate(self.dataloader):
                 -> BaseDataset.__getitem__(...)
                    -> data = BaseDataset.prepare_data(idx)
                    ->
                        data_info = self.get_data_info(idx)
                        ->
                            {
                                'img_path': self.data_prefix['img'] / img_info['file_name']
                                'img_id': img_info['id'],
                                'seg_map_path': None,
                                'height': img_info['height'],
                                'width': img_info['width'],
                                'instances': [
                                                {
                                                    'ignore_flag': 1 if iscrowd else 0,
                                                    'bbox': [x1, y1, x2, y2],
                                                    'bbox_label': category index,
                                                    # If segmentation exists:
                                                    'mask': annotation['segmentation'], 
                                                            # shape: [1, 2 * num_points]
                                                }
                                             ]
                                'sample_idx': idx
                            }

                        return self.pipeline(data_info)
                        ->
                            {
                                'inputs': img, tensor, shape: [3, H, W] BGR,
                                'data_samples': data_sample, DetDataSample instance
                                                META INFORMATION
                                                ori_shape: (427, 640)
                                                img_shape: (800, 1199)
                                                scale_factor: (1.8734375, 1.873536299765808)
                                                img_id: 397133
                                                img_path: 'data/coco/val2017/000000397133.jpg'

                                                DATA FIELDS
                                                gt_instances: <InstanceData(

                                                        META INFORMATION

                                                        DATA FIELDS
                                                        bboxes: HorizontalBoxes(tensor)
                                                        labels: tensor

                                                ignored_instances: <InstanceData(

                                                        META INFORMATION

                                                        DATA FIELDS
                                                        bboxes: HorizontalBoxes(
                                                            tensor([], size=(0, 4)))
                                                        labels: tensor([], dtype=torch.int64)
                                                    )
                            }
                   
                   data_batch = pseudo_collate(...)
                
                # data_batch = {'inputs': [tensor shape [3, H, W]], 'data_samples': [DetDataSample inst]}
                self.run_iter(idx, data_batch)
                -> TestLoop.run_iter(idx, data_batch):
                    outputs = self.runner.model.test_step(data_batch)
                    ->
                        # DDQDETR: DINO: DeformableDETR: DetectionTransformer: BaseDetector: BaseModel
                        #                : BaseModule
                        BaseModel.test_step(data_batch)
                        ->
                            data = self.data_preprocessor(data, False)
                            ->
                                # DetDataPreprocessor: ImgDataPreprocessor: BaseDataPreprocessor
                                DetDataPreprocessor.forward(data, False)
                                -> 
                                    ImgDataPreprocessor.forward(data, False)
                                    # Each tensor of data['inputs']: 
                                    #    bgr_to_rgb
                                    #     Normalization: (tensor - self.mean) / self.std
                                    # stack_batch: pad (bottom, right) + tensors to batch
                                    # Outputs: 
                                    #     data = {
                                    #                'inputs': [tensor shape [B, 3, H, W], 
                                    #                'data_samples': [DetDataSample inst]
                                    #            }
                                    
                                    For each data_sample in data_samples:
                                        data_sample.set_metainfo({
                                            'batch_input_shape': batch_input_shape=(H, W),
                                            'pad_shape': pad_shape=(H_i, W_i) # may != batch_input_shape
                                        })
                                        
                                    samplelist_boxtype2tensor(data_samples)
                            
                            return self._run_forward(data, mode='predict')
                            -> BaseModel._run_forward(data, mode='predict')
                            -> results = self(**data, mode='predict')
                            -> BaseDetector.forward(inputs=inputs, data_samples=data_samples, 
                                                    mode='predict')
                            -> return self.predict(inputs, data_samples)
                            -> DetectionTransformer.predict(batch_inputs, batch_data_samples, rescale=True)
                            -> 
                            	DetectionTransformer.extract_feat(batch_inputs)
                            	
                            	DINO.forward_transformer(img_feats, batch_data_samples)
                            	-> 
                            		self.pre_transformer(img_feats, batch_data_samples)
                            		-> DeformableDETR.pre_transformer
                            		
                            		self.forward_encoder(**encoder_inputs_dict)
                            		-> DeformableDETR.forward_encoder
                            			# DeformableDetrTransformerEncoder
                            		
                            		self.pre_decoder(...)
                            		-> DDQDETR.pre_decoder
                            		
                            		self.forward_decoder(**decoder_inputs_dict)
                            		-> DINO.forward_decoder(...)
                            		-> DDQTransformerDecoder.forward(...)
                            			# DeformableDetrTransformerDecoderLayer
                            			
                            	
                            	# DDQDETRHead
                            	self.bbox_head.predict(**head_inputs_dict,
                                                        rescale=rescale,
                                                        batch_data_samples=batch_data_samples)
                                -> DeformableDETRHead.predict
                                	-> DDQDETRHead.predict_by_feat
                                
                    self.evaluator.process(data_samples=outputs, data_batch=data_batch)

            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
```



Model Class Architecture

| Class                | Methods |      |
| -------------------- | ------- | ---- |
| DDQDETR              |         |      |
| DINO                 |         |      |
| DeformableDETR       |         |      |
| DetectionTransformer |         |      |
| BaseDetector         |         |      |
| BaseModel            |         |      |
| BaseModule           |         |      |



| Key               | Runner Property   | Default Value   |
| ----------------- | ----------------- | --------------- |
| model             |                   | cfg['model']    |
| work_dir          | _work_dir         | cfg['work_dir'] |
| train_dataloader  | _train_dataloader |                 |
| val_dataloader    | _val_dataloader   |                 |
| test_dataloader   | _test_dataloader  |                 |
| train_cfg         | _train_loop       |                 |
| val_cfg           | _val_loop         |                 |
| test_cfg          | _test_loop        |                 |
| auto_scale_lr     | auto_scale_lr     |                 |
| optim_wrapper     | optim_wrapper     |                 |
| param_scheduler   | param_schedulers  |                 |
| val_evaluator     | _val_evaluator    |                 |
| test_evaluator    | _test_evaluator   |                 |
| default_hooks     |                   |                 |
| custom_hooks      |                   |                 |
| data_preprocessor |                   |                 |
| load_from         | _load_from        |                 |
| resume            | _resume           | False           |
| launcher          | _launcher         | 'none'          |
| env_cfg           |                   |                 |
| log_processor     |                   |                 |
| log_level         |                   | 'INFO'          |
| visualizer        |                   |                 |
| default_scope     |                   | 'mmengine'      |
| randomness        |                   | dict(seed=None) |
| experiment_name   |                   |                 |
| cfg               | cfg               | cfg             |





## Runner

### Runner.init

```
mmengine/runner/runner.py

class Runner:
    def __init__(...):
        self.message_hub = self.build_message_hub()
        
        # self.model: DDQDETR instance.
        self.model = self.build_model(model)
        
        # self.model: MMDistributedDataParallel instance.
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)
        
        self.register_hooks(default_hooks, custom_hooks)
        
        
```



### Runner.build_model

```
mmengine/runner/runner.py

class Runner:
    def build_model(self, model):
        model = MODELS.build(model)
        return model
```





### Runner.wrap_model

```
mmengine/runner/runner.py

class Runner:
    def wrap_model(...):
        model = model.to(get_device())
        model = MMDistributedDataParallel(module=model, ...)
        return model
```



### Runner.test

```
mmengine/runner/runner.py

class Runner:
    def test(self) -> dict:
        self._test_loop = self.build_test_loop(self._test_loop)

        self.load_or_resume()

        metrics = self.test_loop.run()

        return metrics
```





### Runner.build_test_loop

```
mmengine/runner/runner.py

class Runner:
    def build_test_loop(self, loop):
        loop_cfg = copy.deepcopy(loop)
        
        loop = LOOPS.build(
            loop_cfg,
            default_args=dict(
                runner=self,
                dataloader=self._test_dataloader,
                evaluator=self._test_evaluator))
        return loop
```





### Runner.build_dataloader

```
mmengine/runner/runner.py

class Runner:
    def build_dataloader(dataloader, seed, diff_rank_seed):
        dataloader_cfg = copy.deepcopy(dataloader)
        
        dataset_cfg = dataloader_cfg.pop('dataset')
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
                
        sampler_cfg = dataloader_cfg.pop('sampler')
        sampler_seed = None if diff_rank_seed else seed
        sampler = DATA_SAMPLERS.build(
            sampler_cfg,
            default_args=dict(dataset=dataset, seed=sampler_seed))
            
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler,
                batch_size=dataloader_cfg.pop('batch_size')))
        
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader
```





### Runner.build_evaluator

```
mmengine/runner/runner.py

class Runner:
    def build_evaluator(self, evaluator):
        return Evaluator(evaluator)
```



## TestLoop

### TestLoop.init

```
mmengine\runner\loops.py

class TestLoop(BaseLoop):
    def __init__(...):
        super().__init__(runner, dataloader)
        self.evaluator = runner.build_evaluator(evaluator)
        
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo

```



### TestLoop.run

```
mmengine\runner\loops.py

class TestLoop(BaseLoop):
    def run(self):
        self.runner.model.eval()
        
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        
        return metrics
```



### TestLoop.run_iter

```
class TestLoop(BaseLoop):
    def run_iter(self, idx, data_batch):
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)

```





## BaseLoop

### BaseLoop.init

```
class BaseLoop(metaclass=ABCMeta):
    def __init__(self, runner, dataloader):
        self.dataloader = runner.build_dataloader(
            dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
```



## Evaluator

### Evaluator.init

```
class Evaluator:
    def __init__(self, metrics):
        # [METRICS.build(metric) for each metric in metrics or [metrics]]
        self.metrics
```



## CocoMetric

### CocoMetric.init

```
mmdet\evaluation\metrics\coco_metric.py

class CocoMetric(BaseMetric):
    def __init__(...):
```





## MODELS

```
mmengine\registry\root.py

MODELS = Registry('model', build_model_from_cfg)
```



# DDQDETR Model Init

 Config file:  `configs\ddq_detr\ddq-detr-4scale_r50_8xb2-12e_coco.py`



## Init Order

| DDQDETR  | DINO                 | DeformableDETR       | DetectionTransformer | BaseDetector | BaseModel          | BaseModule |
| -------- | -------------------- | -------------------- | -------------------- | ------------ | ------------------ | ---------- |
| decoder  |                      |                      |                      |              |                    |            |
| dqs_cfg  |                      |                      |                      |              |                    |            |
|          | dn_cfg               |                      |                      |              |                    |            |
|          |                      | with_box_refine      |                      |              |                    |            |
|          |                      | as_two_stage         |                      |              |                    |            |
|          |                      | num_feature_levels=4 |                      |              |                    |            |
|          |                      |                      |                      |              | data_preprocessor* |            |
|          |                      |                      |                      |              |                    | init_cfg   |
|          |                      |                      | backbone*            |              |                    |            |
|          |                      |                      | neck*                |              |                    |            |
|          |                      |                      | encoder              |              |                    |            |
|          |                      |                      | decoder              |              |                    |            |
|          |                      |                      | bbox_head*           |              |                    |            |
|          |                      |                      | positional_encoding  |              |                    |            |
|          |                      |                      | num_queries          |              |                    |            |
|          |                      |                      | train_cfg            |              |                    |            |
|          |                      |                      | test_cfg             |              |                    |            |
|          |                      |                      |                      |              |                    |            |
|          |                      |                      |                      |              |                    |            |
|          | positional_encoding* |                      |                      |              |                    |            |
|          | encoder*             |                      |                      |              |                    |            |
|          | decoder*             |                      |                      |              |                    |            |
| decoder* |                      |                      |                      |              |                    |            |
|          | dn_cfg*              |                      |                      |              |                    |            |





## DDQDETR

### DDQDETR.init

```
projects\models\ddq_detr.py

class DDQDETR(DINO):
    def __init__(...):
        self.decoder_cfg = kwargs['decoder']
        self.dqs_cfg = dqs_cfg
        
        super().__init__(*args, **kwargs)
        # -> DINO.init
            super().__init__(*args, **kwargs)
            # -> DeformableDETR.init
                self.with_box_refine = with_box_refine
                self.as_two_stage = as_two_stage
                self.num_feature_levels = num_feature_levels
                
                bbox_head['share_pred_layer'] = not with_box_refine
                bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
                    if self.as_two_stage else decoder['num_layers']
                bbox_head['as_two_stage'] = as_two_stage

                super().__init__(*args, decoder=decoder, bbox_head=bbox_head, **kwargs)
                # -> DetectionTransformer.init
                    super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
                    # -> BaseDetector.init
                        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
                        # -> BaseModel.init
                            super().__init__(init_cfg)
                            # -> BaseModule.init
                                super().__init__()
                                self._is_init = False
                                self.init_cfg = copy.deepcopy(init_cfg)
                            
                            self.data_preprocessor = MODELS.build(data_preprocessor)
                            # -> DetDataPreprocessor.init
                    
                    bbox_head.update(train_cfg=train_cfg)
                    bbox_head.update(test_cfg=test_cfg)
                    self.train_cfg = train_cfg
                    self.test_cfg = test_cfg
                    self.encoder = encoder
                    self.decoder = decoder
                    self.positional_encoding = positional_encoding
                    self.num_queries = num_queries

                    # init model layers
                    self.backbone = MODELS.build(backbone)
                    if neck is not None:
                        self.neck = MODELS.build(neck)
                    self.bbox_head = MODELS.build(bbox_head)
                    self._init_layers()
            
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
            self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        
        cache_dict = dict()
        for m in self.modules():
            m.cache_dict = cache_dict
        
        # Set self.cache_dict

```





## DINO

### DINO.init

```
mmdet\models\detectors\dino.py

class DINO(DeformableDETR):
    def __init__(...):
        super().__init__(*args, **kwargs)
        
        dn_cfg['num_classes'] = self.bbox_head.num_classes
        dn_cfg['embed_dims'] = self.embed_dims
        dn_cfg['num_matching_queries'] = self.num_queries
        
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
```



## DeformableDETR

### DeformableDETR.init

```


class DeformableDETR(DetectionTransformer):
    def __init__(...):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        
        bbox_head['share_pred_layer'] = not with_box_refine
        bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
            if self.as_two_stage else decoder['num_layers']
        bbox_head['as_two_stage'] = as_two_stage

        super().__init__(*args, decoder=decoder, bbox_head=bbox_head, **kwargs)

```



## DetectionTransformer

### DetectionTransformer.init

```
D:\proj\git\gitee_DDQ\mmdet\models\detectors\base_detr.py

class DetectionTransformer(BaseDetector, metaclass=ABCMeta):
    def __init__(...):
        
```



## BaseDetector

### BaseDetector.init

```
mmdet\models\detectors\base.py

class BaseDetector(BaseModel, metaclass=ABCMeta):
    def __init__(self, data_preprocessor, init_cfg):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
```





```
mmdet\models\detectors\base.py

class BaseDetector(BaseModel, metaclass=ABCMeta):
    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
                
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # -> DetectionTransformer.predict
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

```





## BaseModel

BaseModel.init

```
mmengine\model\base_model\base_model.py

class BaseModel(BaseModule):
    def __init__(self, data_preprocessor, init_cfg):
        super().__init__(init_cfg)
```



### BaseModel.test_step

```
mmengine\model\base_model\base_model.py

class BaseModel(BaseModule):
    def test_step(self, data):
        data = self.data_preprocessor(data, False)
        
        # -> BaseDetector.forward
        return self._run_forward(data, mode='predict')
```





```
mmengine\model\base_model\base_model.py

class BaseModel(BaseModule):
    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
```





## BaseModule

### BaseModule.init

```
mmengine\model\base_module.py

class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super().__init__()
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)
        
```



## ResNet

### ResNet.init

```
mmdet\models\backbones\resnet.py

class ResNet(BaseModule):
    def __init__(...):
        # Inputs:
            depth=50
        #
        
        # self.block=Bottleneck
        # stage_blocks=(3, 4, 6, 3)
        self.block, stage_blocks = self.arch_settings[depth]
        # in_channels=3
        # stem_channels=64
        self._make_stem_layer(in_channels, stem_channels)
        
        for i, num_blocks in enumerate(self.stage_blocks):
            # self.inplanes: input channel
            # planes: Bottleneck mid channel
            # output channel = planes * 4
            res_layer = self.make_res_layer(...)
```



### ResNet._make_stem_layer

```
mmdet\models\backbones\resnet.py

class ResNet(BaseModule):
    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```



### ResNet.forward

```
mmdet\models\backbones\resnet.py

class ResNet(BaseModule):
    def forward(self, x):
        # self.conv1 = build_conv_layer(...)
        # nn.Conv2d(in_channels=3, out_channels=stem_channels=64, kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False)
        #
        x = self.conv1(x)
        
        # self.norm1 returns self.bn1 = build_norm_layer(...)
        # nn.BatchNorm2d(num_features=64, eps=1e-5)
        # for param in self.norm1.parameters():
        #     param.requires_grad = False
        x = self.norm1(x)
        
        # nn.ReLU(inplace=True)
        x = self.relu(x)
        
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
```





## weight_init

## Pretrained

```
mmengine\model\weight_init.py

@WEIGHT_INITIALIZERS.register_module(name='Pretrained')
class PretrainedInit:
        def __call__(self, module):
            load_checkpoint(
                module,
                self.checkpoint,
                map_location=self.map_location,
                strict=False,
                logger='current')
```



## load_checkpoint

```
mmengine\runner\checkpoint.py

def load_checkpoint(...):
    # Returns state_dict.
     -> return CheckpointLoader.load_checkpoint(filename, map_location, logger)
     -> checkpoint_loader = cls._get_checkpoint_loader(filename)
          return checkpoint_loader(filename, map_location)
    
     -> @CheckpointLoader.register_scheme(prefixes=('modelzoo://', 'torchvision://'))
        def load_from_torchvision(filename, map_location=None)            
            # ->
            # If torchvision.__version__ < '0.13.0a0', then
            #     find resnet URL from torchvision.models.resnet.model_urls
            model_urls = get_torchvision_models()
              
              return load_from_http(model_urls[model_name], map_location=map_location)
              
     -> @CheckpointLoader.register_scheme(prefixes=('http://', 'https://'))
         def load_from_http(...)
     -> checkpoint = load_url(
            filename,
            model_dir=model_dir,
            map_location=map_location,
            progress=progress)
     -> from torch.utils.model_zoo import load_url
    #
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    
    # -> load_state_dict(model, state_dict, strict, logger)
    return _load_checkpoint_to_model(model, checkpoint, strict, logger,
                                     revise_keys)
```





# Inference



## DDQ DETR



### DetectionTransformer.predict

```
D:\proj\git\gitee_DDQ\mmdet\models\detectors\base_detr.py

class DetectionTransformer(BaseDetector, metaclass=ABCMeta):
    def predict(self,
                batch_inputs,
                batch_data_samples,
                rescale: bool = True):
        # Inputs:
            batch_inputs: [B, C=3, H, W], RGB
            batch_data_samples: [DetDataSample instance]
                DetDataSample instance:
                    META INFORMATION
                    ori_shape: (427, 640)
                    img_shape: (800, 1199)
                    scale_factor: (1.8734375, 1.873536299765808)
                    img_id: 397133
                    img_path: 'data/coco/val2017/000000397133.jpg'
                    'batch_input_shape': (H, W),
                    'pad_shape': pad_shape=(H_i, W_i) # may != batch_input_shape

                    DATA FIELDS
                    gt_instances: <InstanceData(

                            META INFORMATION

                            DATA FIELDS
                            bboxes: tensor
                            labels: tensor

                    ignored_instances: <InstanceData(

                            META INFORMATION

                            DATA FIELDS
                            bboxes: tensor([], size=(0, 4))
                            labels: tensor([], dtype=torch.int64)
                        )
        #
        
        # Inputs: 
            batch_inputs, shape: [B, C=3, H, W]
        # Outputs: 
            img_feats, shape: (
                                [B, 256, H /  8, W /  8],
                                [B, 256, H / 16, W / 16],
                                [B, 256, H / 32, W / 32],
                                [B, 256, H / 64, W / 64]
                               )
        #
        img_feats = self.extract_feat(batch_inputs)
        -> DetectionTransformer.extract_feat(batch_inputs)
        
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        -> DINO.forward_transformer(img_feats, batch_data_samples)
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples)
            -> # Get flatten features, masks, and position embeddings.
               DeformableDETR.pre_transformer(img_feats, batch_data_samples)
                masks = 1 at padding positions else 0, 
                        shape: [B, H, W]
                mlvl_masks, mask resize, list, 
                        shape: [ [B, H_level_i, W_level_i], ...]
                mlvl_pos_embeds, SinePositionalEncoding inst(mlvl_mask), list, 
                        shape: [ [B, C=256, H_level_i, W_level_i], ...]
                        
                For each level:
                    # self.level_embed, shape: [4, C=256]
                    lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                    
                feat_flatten, shape: [B, num_anchors, C]
                lvl_pos_embed_flatten, shape: [B, num_anchors, C]
                mask_flatten, shape: [B, num_anchors]
                spatial_shapes, H_level_i, W_level_i, shape: [num_levels=4, 2]
                level_start_index, start index of anchors for each level, shape: [num_levels=4]
                valid_ratios, nomask_W_level_i / W_level_i, nomask_H_level_i / H_level_i,  
                                shape: [B, num_levels=4, 2]
                                
                encoder_inputs_dict = dict(
                    feat=feat_flatten,                      shape: [B, num_anchors, C]
                    feat_mask=mask_flatten,                 shape: [B, num_anchors]
                    feat_pos=lvl_pos_embed_flatten,         shape: [B, num_anchors, C]
                    spatial_shapes=spatial_shapes,          shape: [num_levels=4, 2]
                    level_start_index=level_start_index,    shape: [num_levels=4]
                    valid_ratios=valid_ratios)              shape: [B, num_levels=4, 2]
                decoder_inputs_dict = dict(
                    memory_mask=mask_flatten,               shape: [B, num_anchors]
                    spatial_shapes=spatial_shapes,          shape: [num_levels=4, 2]
                    level_start_index=level_start_index,    shape: [num_levels=4]
                    valid_ratios=valid_ratios)              shape: [B, num_levels=4, 2]

            encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
            -> DeformableDETR.forward_encoder(**encoder_inputs_dict)
                memory = self.encoder(
                    query=feat,
                    query_pos=feat_pos,
                    key_padding_mask=feat_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios)
                ->
                    # DeformableDetrTransformerEncoder.forward
                    reference_points = self.get_encoder_reference_points(
                        spatial_shapes, valid_ratios, device=query.device)
                    reference_points: Normalized anchor center position (x, y) 
                        according to nomask rate update to target level.
                        shape: [B, num_anchors, num_levels, 2]
                    
                    for layer in self.layers:
                        query = layer(
                            query=query,                         shape: [B, num_anchors, C]
                            query_pos=query_pos,                 shape: [B, num_anchors, C]
                            key_padding_mask=key_padding_mask,   shape: [B, num_anchors]
                            spatial_shapes=spatial_shapes,       shape: [num_levels=4, 2]
                            level_start_index=level_start_index, shape: [num_levels=4]
                            valid_ratios=valid_ratios,           shape: [B, num_levels=4, 2]
                            reference_points=reference_points,   shape: [B, num_anchors, num_levels, 2]
                            **kwargs)
                        -> DeformableDetrTransformerEncoderLayer.forward
                        -> DetrTransformerEncoderLayer.forward
                        -> MultiScaleDeformableAttention.forward
                            value = linear(value)
                            mask value: set 0 where key_padding_mask=1
                            return shape: [B, num_anchors, C]
                    
                    return query
                
                encoder_outputs_dict = dict(
                    memory=memory,                         shape: [B, num_anchors, C]
                    memory_mask=feat_mask,                 shape: [B, num_anchors]
                    spatial_shapes=spatial_shapes)         shape: [num_levels=4, 2]
                    
                return encoder_outputs_dict

            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples)
            -> DDQDETR.pre_decoder(**encoder_outputs_dict, batch_data_samples=batch_data_samples)
            -> 
                output_memory, output_proposals = self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
                    # DeformableDETR.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)
                    For each level:
                        grid: [center_x, center_y] for each anchor, 
                                normalized by nomasked W_level and H_level, 
                                shape: [B, H_level, W_level, 2]
                        proposal: [center_x, center_y, w=0.05 * (2^level), h=0.05 * (2^level)] 
                                for each anchor, 
                                center_x, center_y are normalized by nomasked W_level and H_level, 
                                shape: [B, H_level * W_level, 4]
                    output_proposals, shape: [B, num_anchors, 4]
                    output_proposals_valid, 0.01 < output_proposals < 0.99, anchor center is valid,
                                shape: [B, num_anchors, 1]
                                
                    output_proposals, (center_x, center_y, w_level, h_level), 
                        shape: [B, num_anchors, 4]
                        inversed sigmoid,
                        if position is padding masked or anchor center is not valid, then set inf.

                    output_memory, shape: [B, num_anchors, C]
                        memory + if position is padding masked or anchor center is not valid, then set 0
                        + Linear + Layer norm. 

                    return output_memory, output_proposals
            
                # class score logit, shape: [B, num_anchors, num_classes]
                enc_outputs_class = self.bbox_head.cls_branches[
                    self.decoder.num_layers](
                        output_memory)
                        
                # cxcywh logit, shape: [B, num_anchors, 4]
                enc_outputs_coord_unact = self.bbox_head.reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals
                    
                proposals: sigmoid(enc_outputs_coord_unact), + cxcywh to xyxy, 
                            shape: [B, num_anchors, 4]
                scores: enc_outputs_class.max(-1)[0].sigmoid(), max class score, 
                            shape: [B, num_anchors]
                
                for img_id in range(num_imgs):
                    single_proposals = proposals[img_id]
                    single_scores = scores[img_id]
                    
                    # Returns:
                        boxes, nms remained bbox + score, shape: [num_nms, 5] 
                        keep,  nms remained bbox index,   shape: [num_nms]
                    #
                    _, keep_idxs = batched_nms(single_proposals,      shape: [num_anchors, 4]
                                               single_scores,         shape: [num_anchors]
                                               torch.ones(len(single_scores)), shape: [num_anchors]
                                               self.cache_dict['dqs_cfg']),    dict
                    
                    # class score logit, shape: [min(num_nms, num_queries=900), num_classes]
                    topk_score.append(enc_outputs_class[img_id][keep_idxs][:topk])
                    
                    # cxcywh logit, shape: [min(num_nms, num_queries=900), 4]
                    topk_coords_unact.append(
                        enc_outputs_coord_unact[img_id][keep_idxs][:topk])
                    
                    # Linear, shape: [num_anchors, C]
                    map_memory = self.query_map(memory[img_id].detach())
                    
                    # shape: [min(num_nms, num_queries=900), C]
                    query.append(map_memory[keep_idxs][:topk])
                    
                # Pad each tensor to num_queries.
                # class score logit, shape: [B, num_queries=900, num_classes]
                topk_score = align_tensor(topk_score, topk)
                
                # cxcywh logit, shape: [B, num_queries=900, 4]
                topk_coords_unact = align_tensor(topk_coords_unact, topk)
                
                # shape: [B, num_queries=900, C]
                query = align_tensor(query, topk)
                
                # cxcywh sigmoid, shape: [B, num_queries=900, 4]
                topk_anchor = topk_coords_unact.sigmoid()
                
                # detached cxcywh logit, shape: [B, num_queries=900, 4]
                topk_coords_unact = topk_coords_unact.detach()
                
                self.cache_dict['dis_query_info'] = [0, topk]
                
                # detached cxcywh logit, shape: [B, num_queries=900, 4]
                reference_points = topk_coords_unact
                
                dn_mask, dn_meta = None, None
                
                # detached cxcywh sigmoid, shape: [B, num_queries=900, 4]
                reference_points = reference_points.sigmoid()

                decoder_inputs_dict = dict(query=query,          shape: [B, num_queries=900, C]
                                           memory=memory,        shape: [B, num_anchors, C]
                                           reference_points=reference_points, 
                                                                    detached cxcywh sigmoid,
                                                                    shape: [B, num_queries=900, 4]
                                           dn_mask=dn_mask)      None
                                           
                head_inputs_dict = dict(enc_outputs_class=topk_score,
                                        enc_outputs_coord=topk_anchor,
                                        dn_meta=dn_meta) if self.training else dict()

                return decoder_inputs_dict, head_inputs_dict


            decoder_inputs_dict.update(tmp_dec_in)
            ->
                decoder_inputs_dict = dict(
                    query=query,                            shape: [B, num_queries=900, C]
                    memory=memory,                          shape: [B, num_anchors, C]
                    reference_points=reference_points,      detached cxcywh sigmoid,
                                                            shape: [B, num_queries=900, 4]
                    dn_mask=dn_mask,                        None
                    memory_mask=mask_flatten,               shape: [B, num_anchors]
                    spatial_shapes=spatial_shapes,          shape: [num_levels=4, 2]
                    level_start_index=level_start_index,    shape: [num_levels=4]
                    valid_ratios=valid_ratios)              shape: [B, num_levels=4, 2]

            decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
            -> DINO.forward_decoder(**decoder_inputs_dict)
                inter_states, references = self.decoder(
                    query=query,                            shape: [B, num_queries=900, C]
                    value=memory,                           shape: [B, num_anchors, C]
                    key_padding_mask=memory_mask,           shape: [B, num_anchors]
                    self_attn_mask=dn_mask,                 None
                    reference_points=reference_points,      detached cxcywh sigmoid,
                                                            shape: [B, num_queries=900, 4]
                    spatial_shapes=spatial_shapes,          shape: [num_levels=4, 2]
                    level_start_index=level_start_index,    shape: [num_levels=4]
                    valid_ratios=valid_ratios,              shape: [B, num_levels=4, 2]
                    reg_branches=self.bbox_head.reg_branches)
                
                -> DDQTransformerDecoder.forward
                	intermediate = []
        			intermediate_reference_points = [reference_points]
        
                    self_attn_mask, shape: [num_queries, num_queries]
                    self_attn_mask, shape: [B * num_heads, num_queries, num_queries]
                    
                    for each layer:
                        # reference_points * valid_ratios, 
                        # detached cxcywh sigmoid * nomask ratio at each level,
                        reference_points_input, [B, num_queries, num_levels=4, 4]

                        # shape: [B, num_queries=900, 2C]
                        query_sine_embed = coordinate_to_encoding(
                            reference_points_input[:, :, 0, :], 
                                            # detached cxcywh sigmoid * nomask ratio at level 0, 
                                            # shape: [B, num_queries, 4]
                            num_feats=C // 2)

                        # Input shape:  [B, num_queries=900, 2C]
                        # Output shape: [B, num_queries=900, C]
                        query_pos = self.ref_point_head(query_sine_embed)

                        query = layer(query,                             shape: [B, num_queries=900, C]
                                      query_pos=query_pos,               shape: [B, num_queries=900, C]
                                      value=value,                       shape: [B, num_anchors, C]
                                      key_padding_mask=key_padding_mask, shape: [B, num_anchors]
                                      self_attn_mask=self_attn_mask, 
                                                        shape: [B * num_heads, num_queries, num_queries]
                                      spatial_shapes=spatial_shapes,     shape: [num_levels=4, 2]
                                      level_start_index=level_start_index, shape: [num_levels=4]
                                      valid_ratios=valid_ratios,         shape: [B, num_levels=4, 2]
                                      reference_points=reference_points_input, 
                                                      detached cxcywh sigmoid * nomask ratio at each level,
                                                      shape: [B, num_queries, num_levels=4, 4]
                                      **kwargs)
                        -> DeformableDetrTransformerDecoderLayer.forward
                        -> DetrTransformerDecoderLayer.forward
                                self.self_attn = MultiheadAttention
                                self.cross_attn = MultiScaleDeformableAttention
                        
                        new_reference_points: cxcywh sigmoid, shape: [B, num_queries=900, 4]
                        reference_points = new_reference_points.detach()
                        if lid < (len(self.layers) - 1):
                        	# Inputs:
                                reference_points, shape: [B,             num_queries=900,               4]
                                query,            shape: [B,             num_queries=900,               C]
                                self_attn_mask,   shape: [B * num_heads, num_queries=900, num_queries=900]
                            self_attn_mask = self.be_distinct(reference_points, query,
                                                              self_attn_mask, lid)
                        if self.return_intermediate:
                            intermediate.append(self.norm(query)), shape: [B, num_queries=900, C]
                            intermediate_reference_points.append(new_reference_points),
                                                                    shape: [B, num_queries=900, 4]
                                                                    
                    if self.return_intermediate:
                        return torch.stack(intermediate), 
                        								shape: [num_dec_layers,   B, num_queries=900, C]
                        	torch.stack(intermediate_reference_points), 
                        								shape: [num_dec_layers+1, B, num_queries=900, 4]
                
                
                inter_states,     shape: [num_dec_layers,   B, num_queries=900, C]
                inter_references, shape: [num_dec_layers+1, B, num_queries=900, 4]
                
                # hidden_states,      shape: [num_dec_layers,   B, num_queries=900, C]
                # references, list of shape: [B, num_queries=900, 4]
                #			  len: num_dec_layers+1
                decoder_outputs_dict = dict(hidden_states=inter_states, references=list(references))

                return decoder_outputs_dict
            
            head_inputs_dict.update(decoder_outputs_dict)
            return head_inputs_dict
        
        
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        -> DDQDETRHead.predict(**head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples)
        -> DeformableDETRHead.predict
        	# Inputs:
                **head_inputs_dict: 
                    # hidden_states,      shape: [num_dec_layers,   B, num_queries=900, C]
                    # references, list of shape: [B, num_queries=900, 4]
                    #			  len: num_dec_layers+1
                batch_data_samples: [DetDataSample instance]
                rescale: True
            #
            
            # List of metainfo dict.
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
			
            outs = self(hidden_states, references)
            -> DDQDETRHead.forward(hidden_states, references)
            	# Inputs: 
                    hidden_states,      shape: [num_dec_layers,   B, num_queries=900, C]
                    references, list of shape: [B, num_queries=900, 4]
                                  len: num_dec_layers+1
                # Outputs: Tuple,
                	all_layers_outputs_classes: class score logits,
                		shape: [num_dec_layers,   B, num_queries=900, num_class]
                	all_layers_outputs_coords:  sigmoid cxcywh,
                		shape: [num_dec_layers,   B, num_queries=900, 4]
                #

            predictions = self.predict_by_feat(
                *outs, batch_img_metas=batch_img_metas, rescale=rescale)
            -> DDQDETRHead.predict_by_feat:
            	# Inputs:
            		layer_cls_scores: class score logits,
                		shape: [num_dec_layers,   B, num_queries=900, num_class]
            		layer_bbox_preds: sigmoid cxcywh,
                		shape: [num_dec_layers,   B, num_queries=900, 4]
            		batch_img_metas: List of metainfo dict
            		rescale: True
            	# shape: [B, num_queries=900, num_class]
                cls_scores = layer_cls_scores[-1]
                
                # shape: [B, num_queries=900, 4]
                bbox_preds = layer_bbox_preds[-1]
              	
              	# self.cache_dict['num_heads'][-1], 1 for nomask, 0 for mask
              			shape: [B * num_head, num_dis, num_dis]
              	# batch_mask, List of shape: [num_dis]
              	batch_mask = [
                    self.cache_dict['distinct_query_mask'][-1][
                        img_id * self.cache_dict['num_heads']][0]
                    for img_id in range(num_imgs)
        		]
        		
        		for img_id in range(len(batch_img_metas)):
        			# cls_score, class score logits, shape: [num_dis, num_class]
                    cls_score = cls_scores[img_id][batch_mask[img_id]]
                    
                    # bbox_pred, sigmoid cxcywh, shape: [num_dis, 4]
                    bbox_pred = bbox_preds[img_id][batch_mask[img_id]]
                    
                    # img_meta, metainfo dict
                    img_meta = batch_img_metas[img_id]
                    results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                           img_meta, rescale)
                    -> DETRHead._predict_by_feat_single:
                    	# Inputs:
                    	cls_score, class score logits, shape: [num_dis, num_class]
                    	bbox_pred, sigmoid cxcywh,     shape: [num_dis, 4]
                    	img_meta,  metainfo dict
                    	rescale,   True
						#
						cls_score = cls_score.sigmoid()
						
						Select at most top max_per_img from num_dis * num_class scores, order by cls_score.
						
						Outputs:
						InstanceData inst:
                            det_bboxes, selected bbox_pred + cxcywh to xyxy + denorm to resized image 
                                        	+ clamp + descale to original image,
                                        shape: [max_per_img, 4]
                            scores,     selected sigmoid cls_score,
                                        shape: [max_per_img]
                            det_labels, selected class index,
                                        shape: [max_per_img]
						
              
            return predictions
	        
            
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

```





### DDQTransformerDecoder.be_distinct

```

class DDQTransformerDecoder(DeformableDetrTransformerDecoder):

	def be_distinct(self, ref_points, query, self_attn_mask, lid):
		# Inputs:
        ref_points,       shape: [B,             num_queries=900,               4]
        query,            shape: [B,             num_queries=900,               C]
        self_attn_mask,   shape: [B * num_heads, num_queries=900, num_queries=900]
        lid,              int
        #
        dis_start, num_dis = self.cache_dict['dis_query_info']
        
        dis_mask, from self_attn_mask's distinct range, 
        	shape: [B * num_heads, num_dis, num_dis]
        
        scores, class score sigmoid of max cls_branches(query[distinct range]), 
        	shape: [B, num_dis]
        
        proposals, ref_points[distinct range] + from cxcywh to xyxy, 
        	shape: [B, num_dis, 4]
        
        for img_id in range(num_imgs):
        	ori_index: where dis_mask[(img_id, head_index=0), target position=0] == 0
        				shape: [num_src_nomasked]
        	
        	# keep_idxs, remained index of ori_index after nms, shape: [num_src_nms]
        	_, keep_idxs = batched_nms(...)
        	
        	# Final remained index after nms, shape: [num_src_nms]
        	real_keep_index = ori_index[keep_idxs]
        	
        	attn_mask, set all src or target at real_keep_index 0, + reapeat num_heads.
        	shape: [num_heads, num_dis, num_dis]
        	
       	self_attn_mask = copy.deepcopy(self_attn_mask)
       	self_attn_mask[distinct range] = attn_mask
       	
       	self.cache_dict['distinct_query_mask'].append(~attn_mask)
        
        return self_attn_mask
```



### DINO.forward_transformer

```
mmdet\models\detectors\dino.py


class DINO(DeformableDETR):
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        -> DeformableDETR.pre_transformer
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        
        -> DeformableDETR.forward_encoder
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict
```



## SinePositionalEncoding

### SinePositionalEncoding.forward

Assume image is resized to $ \left (H, W \right ) $ in `train_pipeline` or `test_pipeline`,  `num_feats=128`, then for each `h, w`:
$$
\sin \left ( 2 \pi \cdot \dfrac {h} {H} \cdot { {20} ^{- \dfrac {0} {128} } } \right ), 
\cos \left ( 2 \pi \cdot \dfrac {h} {H} \cdot { {20} ^{- \dfrac {0} {128} } } \right ), 
\sin \left ( 2 \pi \cdot \dfrac {h} {H} \cdot { {20} ^{- \dfrac {2} {128} } } \right ), 
\cos \left ( 2 \pi \cdot \dfrac {h} {H} \cdot { {20} ^{- \dfrac {2} {128} } } \right ), 
\dots,  \\
\sin \left ( 2 \pi \cdot \dfrac {w} {W} \cdot { {20} ^{- \dfrac {0} {128} } } \right ), 
\cos \left ( 2 \pi \cdot \dfrac {w} {W} \cdot { {20} ^{- \dfrac {0} {128} } } \right ), 
\sin \left ( 2 \pi \cdot \dfrac {w} {W} \cdot { {20} ^{- \dfrac {2} {128} } } \right ), 
\cos \left ( 2 \pi \cdot \dfrac {w} {W} \cdot { {20} ^{- \dfrac {2} {128} } } \right ), 
\dots
$$





## MultiScaleDeformableAttention

### MultiScaleDeformableAttention.forward

```


class MultiScaleDeformableAttention(BaseModule):
    def forward(...):
        # Inputs:
            query,              shape: [B, num_queries, C]
            query_pos,          shape: [B, num_queries, C]
            value,              shape: [B, num_anchors, C]
            identity,           None
            key_padding_mask,   shape: [B, num_anchors]
            reference_points,   cxcywh sigmoid * nomask ratio at each level,
                                shape: [B, num_queries, num_levels=4, 2 or 4]
            spatial_shapes,     shape: [num_levels=4, 2]
            level_start_index,  shape: [num_levels=4]
        #
        
        identity = query
        query = query + query_pos
        value = self.value_proj(value)
        value = value.masked_fill(key_padding_mask[..., None], 0.0)
        
        # shape: [B,num_anchors, num_heads, head_dim]
        value = value.view(bs, num_value, self.num_heads, -1)
        
        #  Input shape: [B, num_queries, C]
        # Output shape: [B, num_queries, num_heads, num_levels, num_points, 2]
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        #  Input shape: [B, num_queries, C]
        # Output shape: [B, num_queries, num_heads, num_levels * num_points]
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        
        attention_weights = attention_weights.softmax(-1)
        
        # Output shape: [B, num_queries, num_heads, num_levels, num_points]
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:

            # [W_level, H_level], shape: [num_levels=4, 2]
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                
            # Inputs:
                reference_points[:, :, None, :, None, :]:
                    cxcy sigmoid * nomask ratio at each level,
                    shape: [B, num_queries,         1, num_levels=4,          1, 2]
                sampling_offsets ...:
                    shape: [B, num_queries, num_heads, num_levels=4, num_points, 2]
                offset_normalizer[None, None, None, :, None, :]:
                    shape: [1,           1,         1, num_levels=4,          1, 2]
            # Outputs:
                sampling_locations: 
                    shape: [B, num_queries, num_heads, num_levels=4, num_points, 2]
            #
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # Inputs:
                reference_points[:, :, None, :, None, :2]:
                    cxcy sigmoid * nomask ratio at each level,
                    shape: [B, num_queries,         1, num_levels=4,          1, 2]
                sampling_offsets:
                    shape: [B, num_queries, num_heads, num_levels=4, num_points, 2]
                reference_points[:, :, None, :, None, 2:]:
                    wh sigmoid * nomask ratio at each level,
                    shape: [B, num_queries,         1, num_levels=4,          1, 2]
            # Outputs:
                sampling_locations: 
                    shape: [B, num_queries, num_heads, num_levels=4, num_points, 2]
            #
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        
        # Inputs:
            value:
                shape: [B, num_anchors, num_heads, head_dim]
            spatial_shapes:
                shape: [num_levels=4, 2]
            sampling_locations: 
                shape: [B, num_queries, num_heads, num_levels=4, num_points, 2]
            attention_weights:
                shape: [B, num_queries, num_heads, num_levels, num_points]
        # Outputs:
                shape: [B, num_queries, C]
        #
        output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        
        # Outputs:
                shape: [B, num_queries, C]
        return self.dropout(output) + identity
```



## multi_scale_deformable_attn_pytorch

```

def multi_scale_deformable_attn_pytorch(...):
    # Inputs:
        value:
            shape: [B, num_anchors, num_heads, head_dim]
        value_spatial_shapes:
            shape: [num_levels=4, 2]
        sampling_locations: 
            shape: [B, num_queries, num_heads, num_levels=4, num_points, 2]
        attention_weights:
            shape: [B, num_queries, num_heads, num_levels, num_points]
    # Outputs:
            shape: [B, num_queries, C]
    #
    
    # value_list, list of shape: [B, num_anchors_level, num_heads, head_dim]
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    
    # sampling_grids, location range from [0, 1] to [-1, 1],
    #     shape: [B, num_queries, num_heads, num_levels=4, num_points, 2]
    sampling_grids = 2 * sampling_locations - 1
    
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # value_l_, 
        #    shape: [B * num_heads, head_dim, H_level, W_level]
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
         
        # sampling_grid_l_, 
        #    shape: [B * num_heads, num_queries, num_points, 2]
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        
        # sampling_value_l_,
        #     shape: [B * num_heads, head_dim, num_queries, num_points]
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    
    # attention_weights,
    #     shape: [B * num_heads,          1, num_queries, num_levels * num_points]
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    
    # torch.stack(sampling_value_list, dim=-2).flatten(-2)
    #     shape: [B * num_heads, embed_dims, num_queries, num_levels * num_points]
    # output, shape: [B, C, num_queries]
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    # output, shape: [B, C, num_queries]
    # output.transpose(1, 2), shape: [B, num_queries, C]
    return output.transpose(1, 2).contiguous()
```





## DDQDETRHead

### Init Order

`bbox_head` 

| DDQDETRHead   | DINOHead | DeformableDETRHead         | DETRHead            | BaseModule |
| ------------- | -------- | -------------------------- | ------------------- | ---------- |
| dn_loss=True  |          |                            |                     |            |
| aux_num_pos=4 |          |                            |                     |            |
|               |          | share_pred_layer=False (1) |                     |            |
|               |          | num_pred_layer=7  (1)      |                     |            |
|               |          | as_two_stage=True  (1)     |                     |            |
|               |          |                            | num_classes         |            |
|               |          |                            | embed_dims=256      |            |
|               |          |                            | num_reg_fcs=2       |            |
|               |          |                            | sync_cls_avg_factor |            |
|               |          |                            | loss_cls            |            |
|               |          |                            | loss_bbox           |            |
|               |          |                            | loss_iou            |            |
|               |          |                            | train_cfg (2)       |            |
|               |          |                            | test_cfg (2)        |            |
|               |          |                            | init_cfg=None       | init_cfg   |

 (1) Assigned by `DeformableDETR.__init__`

 (2) Assigned by `DetectionTransformer.__init__`




## CocoDataset

### CocoDataset.load_data_list

```
mmdet\datasets\coco.py

class CocoDataset(BaseDetDataset):
    def load_data_list(self):
        # local_path=self.ann_file 
        self.coco = self.COCOAPI(local_path)
        
        # [category_id for each category in database['categories'] if category in self.metainfo['classes']]
        self.cat_ids
        
        # {category_id: category_index for each category in database['categories'] 
        #    if category in self.CLASSES}
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        # {category_id: [image_id in the category]}
        # COCO.catToImgs
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        
        data_list: [
                    # Data info for each image
                    {
                        'img_path': self.data_prefix['img'] / img_info['file_name']
                        'img_id': img_info['id'],
                        'seg_map_path': None,
                        'height': img_info['height'],
                        'width': img_info['width'],
                        'instances': [
                                        {
                                            'ignore_flag': 1 if iscrowd else 0,
                                            'bbox': [x1, y1, x2, y2],
                                            'bbox_label': category index,
                                            # If segmentation exists:
                                            'mask': annotation['segmentation'], shape: [1, 2 * num_points]
                                        }
                                     ]
                    }
                   ]
        
        return data_list
```





# Data Pipline



| Init         | LoadImageFromFile         | Resize            | LoadAnnotations   | RandomFlip     | RandomChoiceResize | RandomCrop        | PackDetInputs |
| ------------ | ------------------------- | ----------------- | ----------------- | -------------- | ------------------ | ----------------- | ------------- |
| img_path     |                           |                   |                   |                |                    |                   |               |
| img_id       |                           |                   |                   |                |                    |                   |               |
| seg_map_path |                           |                   |                   |                |                    |                   |               |
| height       |                           |                   |                   |                |                    |                   |               |
| width        |                           |                   |                   |                |                    |                   |               |
| instances    |                           |                   |                   |                |                    |                   |               |
| sample_idx   |                           |                   |                   |                |                    |                   |               |
|              | img: shape: [H, W, 3]     | img               |                   | img            | img                | img               | inputs        |
|              | img_shape = img.shape[:2] | img_shape         |                   |                | img_shape          | img_shape         |               |
|              | ori_shape = img.shape[:2] |                   |                   |                |                    |                   |               |
|              |                           | scale             |                   |                | scale              |                   |               |
|              |                           | scale_factor      |                   |                | scale_factor       |                   |               |
|              |                           | keep_ratio        |                   |                | keep_ratio         |                   |               |
|              |                           | homography_matrix | homography_matrix |                | homography_matrix  | homography_matrix |               |
|              |                           |                   | gt_bboxes         | gt_bboxes      |                    | gt_bboxes         |               |
|              |                           |                   | gt_ignore_flags   |                |                    | gt_ignore_flags   |               |
|              |                           |                   | gt_bboxes_labels  |                |                    | gt_bboxes_labels  |               |
|              |                           |                   |                   | flip           |                    |                   |               |
|              |                           |                   |                   | flip_direction |                    |                   |               |
|              |                           |                   |                   |                | scale_idx          |                   |               |
|              |                           |                   |                   |                |                    |                   | data_samples  |



## BaseDataset.get_data_info

```
BaseDataset.get_data_info(self, idx):
    # Returns
    {
        'img_path': self.data_prefix['img'] / img_info['file_name']
        'img_id': img_info['id'],
        'seg_map_path': None,
        'height': img_info['height'],
        'width': img_info['width'],
        'instances': [
                        {
                            'ignore_flag': 1 if iscrowd else 0,
                            'bbox': [x1, y1, x2, y2],
                            'bbox_label': category index,
                            # If segmentation exists:
                            'mask': annotation['segmentation'], 
                                    # shape: [1, 2 * num_points]
                        }
                     ]
        'sample_idx': idx
    }
    #
```





## LoadImageFromFile

```
class LoadImageFromFile(BaseTransform):
    def transform(self, results):
        with open(results['img_path'], 'rb') as f: 
            content = f.read()

        img_np = np.frombuffer(content, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # [H, W, 3], BGR
```







## Resize

```
mmdet\datasets\transforms\transforms.py

class Resize(MMCV_Resize):
    def transform(self, results):
        # If self.scale and self.keep_ratio:
        #    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
        self._resize_img(results)
        
        self._resize_bboxes(results)
        
        self._resize_masks(results)
        
        self._resize_seg(results)
        
        self._record_homography_matrix(results)
        return results
```



## LoadAnnotations

```
class LoadAnnotations(MMCV_LoadAnnotations):
    def transform(self, results):
        if self.with_bbox:
            # results['gt_bboxes'] = HorizontalBoxes(gt_bboxes, dtype=torch.float32)
            #     self.tensor shape: [num_bboxes, 4]
            # results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
            #     shape: [num_bboxes]
            self._load_bboxes(results)
        if self.with_label:
            # results['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=np.int64)
            #     shape: [num_bboxes]
            self._load_labels(results)
```





## PackDetInputs

```


class PackDetInputs(BaseTransform):
    def transform(self, results):
        # Outputs: 
        packed_results= {
                            'inputs': img, tensor, shape: [3, H, W] BGR,
                            'data_samples': data_sample, DetDataSample instance
                        }
        #
        
        data_sample: DetDataSample instance.
        data_sample.gt_instances = instance_data: InstanceData instance
        data_sample.ignored_instances = ignore_instance_data: InstanceData instance
        
               instance_data.bboxes: results['gt_bboxes']=HorizontalBoxes(gt_bboxes, dtype=torch.float32),
                                        shape: [num_bbox_nocrowd, 4]
               instance_data.labels: results['gt_bboxes_labels'], tensor, shape: [num_bbox_nocrowd]
        
        ignore_instance_data.bboxes: results['gt_bboxes']=HorizontalBoxes(gt_bboxes, dtype=torch.float32),
                                        shape: [num_bbox_iscrowd, 4]
        ignore_instance_data.labels: results['gt_bboxes_labels'], tensor, shape: [num_bbox_iscrowd]
        
        # img_meta = {
                        'img_id': img_info['id'] in BaseDataset.get_data_info, 
                        'img_path': BaseDataset.data_prefix['img'] / img_info['file_name'] 
                                        in BaseDataset.get_data_info, 
                        'ori_shape': origion [H, W, 3] in LoadImageFromFile, 
                        'img_shape': [H, W, 3] in Resize,
                        'scale_factor': (W_scale_factor, H_scale_factor) in Resize
                     }
        #
        data_sample.set_metainfo(img_meta)
```







# Train

## train.main



```
python tools/train.py projects/configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py ./exp/ddq_detr

cfg = Config.fromfile(args.config)
cfg.work_dir = args.work_dir

runner = RUNNERS.build(cfg)
# Runner.__init__
class Runner:
    self.model = self.build_model(model)

    self.model = self.wrap_model(self.cfg.get('model_wrapper_cfg'), self.model)


runner.train()
# Runner.train
	self._train_loop = self.build_train_loop(self._train_loop)
    -> EpochBasedTrainLoop.__init__
    	-> BaseLoop.__init__
    		self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            -> Runner.build_dataloader
            	...

	self._val_loop = self.build_val_loop(self._val_loop)

    self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
    self.scale_lr(self.optim_wrapper, self.auto_scale_lr)
    self.param_schedulers = self.build_param_scheduler(self.param_schedulers)
    self._init_model_weights()
    self.load_or_resume()

    model = self.train_loop.run()
    -> EpochBasedTrainLoop.run()
    	while self._epoch < self._max_epochs:
            self.run_epoch()
            ->EpochBasedTrainLoop.run_epoch()
    			self.runner.model.train()
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)
                    -> EpochBasedTrainLoop.run_iter(idx, data_batch)
						outputs = self.runner.model.train_step(
            				data_batch, optim_wrapper=self.runner.optim_wrapper)
            			-> BaseModel.train_step
    						with optim_wrapper.optim_context(self):
                                data = self.data_preprocessor(data, True)
                                losses = self._run_forward(data, mode='loss')
                                -> BaseModel._run_forward(data, mode='loss')
                                -> results = self(**data, mode='loss')
                                -> BaseDetector.forward(inputs=inputs, data_samples=data_samples, 
                                                    	mode='loss')
                            -> return self.loss(inputs, data_samples)
                            -> DetectionTransformer.loss(batch_inputs, batch_data_samples)
                            	# losses, {key: Tensor or List of Tensor}
                            	
                            parsed_losses, log_vars = self.parse_losses(losses)
                            -> 	parsed_losses, 1 tensor value. 
                            					Sum of each tensor.mean() where 'loss' in key.
                            	log_vars, {key: 1 tensor value}
                            
                            optim_wrapper.update_params(parsed_losses)
```





## DetectionTransformer

### DetectionTransformer.loss

```

class DetectionTransformer(BaseDetector, metaclass=ABCMeta):
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        img_feats = self.extract_feat(batch_inputs)
        
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        
        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples)
		
		return losses
```





## DINO



### DINO.forward_transformer

```
DeformableDETR.pre_transformer->
    encoder_inputs_dict = dict(
        feat=feat_flatten,                      shape: [B, num_anchors, C]
        feat_mask=mask_flatten,                 shape: [B, num_anchors]
        feat_pos=lvl_pos_embed_flatten,         shape: [B, num_anchors, C]
        spatial_shapes=spatial_shapes,          shape: [num_levels=4, 2]
        level_start_index=level_start_index,    shape: [num_levels=4]
        valid_ratios=valid_ratios)              shape: [B, num_levels=4, 2]
    decoder_inputs_dict = dict(
        memory_mask=mask_flatten,               shape: [B, num_anchors]
        spatial_shapes=spatial_shapes,          shape: [num_levels=4, 2]
        level_start_index=level_start_index,    shape: [num_levels=4]
        valid_ratios=valid_ratios)              shape: [B, num_levels=4, 2]

DeformableDETR.forward_encoder->
    encoder_outputs_dict = dict(
        memory=memory,                         shape: [B, num_anchors, C]
        memory_mask=feat_mask,                 shape: [B, num_anchors]
        spatial_shapes=spatial_shapes)         shape: [num_levels=4, 2]

DINO.pre_decoder->
    decoder_inputs_dict = dict(
        query=query,                          # shape: [B, num_dino_queries + num_queries, C]
        memory=memory,                        # shape: [B, num_anchors, C]
        reference_points=reference_points,    # shape: [B, num_dino_queries + num_queries, 4]
        									  # sigmoid cxcywh
        dn_mask=dn_mask)

    head_inputs_dict = dict(
        enc_outputs_class=topk_score,         # shape: [B, num_queries, num_classes]
        enc_outputs_coord=topk_coords,        # shape: [B, num_queries, 4], sigmoid cxcywh
        dn_meta=dn_meta)

DDQDETR.pre_decoder->
    decoder_inputs_dict = dict(
    	query=query,                        # shape: [B, num_dino_queries + num_queries + num_dense_queries,
        												C]
        memory=memory,                      # shape: [B, num_anchors, C]
        reference_points=reference_points,  # shape: [B, num_dino_queries + num_queries + num_dense_queries,
        												4],
        									# sigmoid cxcywh
            # detached cxcywh sigmoid,
        dn_mask=dn_mask)

    head_inputs_dict = dict(
        enc_outputs_class=topk_score,            # shape: [B, num_queries, num_classes]
        enc_outputs_coord=topk_coords,           # shape: [B, num_queries, 4], sigmoid cxcywh
        aux_enc_outputs_class=dense_topk_score,  # shape: [B, num_dense_queries, num_classes]
        aux_enc_outputs_coord=dense_topk_coords, # shape: [B, num_dense_queries, 4], sigmoid cxcywh
        dn_meta=dn_meta)


decoder_inputs_dict.update(tmp_dec_in)
->
    decoder_inputs_dict = dict(
        memory_mask=mask_flatten,               shape: [B, num_anchors]
        spatial_shapes=spatial_shapes,          shape: [num_levels=4, 2]
        level_start_index=level_start_index,    shape: [num_levels=4]
        valid_ratios=valid_ratios              shape: [B, num_levels=4, 2]
        query=query,                          # shape: [B, num_dino_queries + num_queries, C]
        memory=memory,                        # shape: [B, num_anchors, C]
        reference_points=reference_points,    # shape: [B, num_dino_queries + num_queries, 4], 
        									  # sigmoid cxcywh
        dn_mask=dn_mask)
        
DINO.forward_decoder
->
	decoder_outputs_dict = dict(
		hidden_states=inter_states,           List of shape: [B, num_dino_queries + num_queries, C]
												# len=num_layers
		references=list(references)),         List of shape: [B, num_dino_queries + num_queries, 4]
												# len=num_layers + 1, sigmoid cxcywh

head_inputs_dict.update(decoder_outputs_dict)
->
    head_inputs_dict = dict(
        enc_outputs_class=topk_score,         # shape: [B, num_queries, num_classes]
        enc_outputs_coord=topk_coords,        # shape: [B, num_queries, 4], sigmoid cxcywh
        dn_meta=dn_meta,
		hidden_states=inter_states,           List of shape: [B, num_dino_queries + num_queries, C]
												# len=num_layers
		references=list(references)),         List of shape: [B, num_dino_queries + num_queries, 4]
												# len=num_layers + 1, sigmoid cxcywh

# Return to DetectionTransformer.loss
# Passed to DINOHead inst.loss 
return head_inputs_dict
```



### DINO.pre_decoder

```
D:\proj\git\mmdetection\mmdet\models\detectors\dino.py

class DINO(DeformableDETR):
    def pre_decoder(...):
    	# Inputs:
            memory=memory,                         shape: [B, num_anchors, C]
            memory_mask=feat_mask,                 shape: [B, num_anchors]
            spatial_shapes=spatial_shapes,         shape: [num_levels=4, 2]
            batch_data_samples,                    [DetDataSample inst]
    	#
    	
    	output_memory,                             shape: [B, num_anchors, C]
    	output_proposals,                          shape: [B, num_anchors, 4]
    	
    	enc_outputs_class,                         shape: [B, num_anchors, num_classes]
    	enc_outputs_coord_unact,                   shape: [B, num_anchors, 4]
    	
    	topk_indices,                              shape: [B, num_queries]
    	topk_score,                                shape: [B, num_queries, num_classes]
    	topk_coords_unact,                         shape: [B, num_queries, 4]
    	
    	query,                                     shape: [B, num_queries, C]
    	
    	# dn_label_query, shape: [B, num_dino_queries, C]
    	# dn_bbox_query,  shape: [B, num_dino_queries, 4]
    	dn_label_query, dn_bbox_query, dn_mask, dn_meta =  self.dn_query_generator(batch_data_samples)

		query = torch.cat([dn_label_query, query], dim=1)
                                                   shape: [B, num_dino_queries + num_queries, C]
        reference_points = torch.cat([dn_bbox_query, topk_coords_unact],  dim=1)
                                                   shape: [B, num_dino_queries + num_queries, 4]
        
        decoder_inputs_dict = dict(
            query=query,                          # shape: [B, num_dino_queries + num_queries, C]
            memory=memory,                        # shape: [B, num_anchors, C]
            reference_points=reference_points,    # shape: [B, num_dino_queries + num_queries, 4]
            dn_mask=dn_mask)
        
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,shape:   # shape: [B, num_queries, num_classes]
            enc_outputs_coord=topk_coords,shape:  # shape: [B, num_queries, 4]
            dn_meta=dn_meta)

return decoder_inputs_dict, head_inputs_dict
```

## CdnQueryGenerator

```
D:\proj\git\mmdetection\mmdet\models\layers\transformer\dino_layers.py

class CdnQueryGenerator(BaseModule):

    def __call__(...):   
    	gt_labels, shape: [num_all_gt_bboxes]
    	gt_bboxes, shape: [num_all_gt_bboxes, 4]
    	num_groups = max(1, num_dn_queries=100 // max_num_gt_bboxes)
    	
        # Replace randomly 1/4 gt labels in gt_labels, and embedding, 
        # shape: [num_groups * 2 * num_all_gt_bboxes, C]
        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)

        # dn_bbox_query: 
            normed gt bbox xyxy, 
            for each left/top/right/bottom side:
                for positive: randomly + (-1 / 2, 1 / 2) bbox_w or bbox_h
                for negative: randomly + (-1, -1 / 2] U [1 / 2, 1) bbox_w or bbox_h
            clamp [0, 1]
            xyxy to cxcywh
            un_sigmoid
        # 	shape: [num_groups * 2 * num_all_gt_bboxes, 4]
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        # image idx for each gt bbox, image_id, shape: [num_all_gt_bboxes]

        # num_target_list: number of gt bboxes in each image

        # dn_label_query, [B, num_groups * 2 * max_num_gt_bboxes, C]
        #  dn_bbox_query, [B, num_groups * 2 * max_num_gt_bboxes, 4]
        dn_label_query, dn_bbox_query = self.collate_dn_queries

        num_queries_total = num_denoising_queries + num_matching_queries
        attn_mask, shape: [num_queries_total, num_queries_total]

        dn_meta = dict(
            num_denoising_queries=int(num_groups * 2 * max_num_gt_bboxes),
            num_denoising_groups=num_groups)
            
		return dn_label_query, dn_bbox_query, attn_mask, dn_meta
```



# Head

## DINOHead

```
class DINOHead(DeformableDETRHead):
    def loss(...):
    	# Inputs:
            hidden_states=inter_states,         # List of shape: [B, num_dino_queries + num_queries, C]
												# len=num_layers
            references=list(references)),       # List of shape: [B, num_dino_queries + num_queries, 4]
												# len=num_layers + 1
            enc_outputs_class=topk_score,       # shape: [B, num_queries, num_classes]
            enc_outputs_coord=topk_coords,      # shape: [B, num_queries, 4]
            									# sigmoid cxcywh
            batch_data_samples, 
            dn_meta
        #
		
		batch_gt_instances, # List of gt_instances for each image.
		batch_img_metas,    # List of metainfo for each image.
		
        outs = self(hidden_states, references)
        -> DeformableDETRHead.forward(hidden_states, references):
	        # Outputs:
        	all_layers_outputs_classes, shape: [num_layers, B, num_dino_queries + num_queries, num_classes]
        	all_layers_outputs_coords,  shape: [num_layers, B, num_dino_queries + num_queries, 4]
        		sigmoid cxcywh
        		
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
        
    def loss_by_feat(...):
    	# Inputs:
    	#
        all_layers_cls_scores, all_layers_outputs_classes, 
        						shape: [num_layers, B, num_dino_queries + num_queries, num_classes]
        all_layers_bbox_preds, all_layers_outputs_coords, sigmoid cxcywh,
        						shape: [num_layers, B, num_dino_queries + num_queries, 4]
        enc_cls_scores, enc_outputs_class=topk_score,       
        						shape: [B, num_queries, num_classes]
        enc_bbox_preds, enc_outputs_coord=topk_coords,      
        						shape: [B, num_queries, 4]
        batch_gt_instances,     List of gt_instances for each image.
        batch_img_metas,        List of metainfo for each image.
        dn_meta,
        batch_gt_instances_ignore = None
    )
```



## DETRHead

```
D:\proj\git\mmdetection\mmdet\models\dense_heads\detr_head.py

class DETRHead(BaseModule):
    def loss_by_feat_single(...):
    	# Calculate loss of output of each encoder / decoder layer from matched queries, not dino queries.
    	# Inputs:
            cls_scores,             shape: [B, num_queries, num_classes]
            bbox_preds, sigmoid cxcywh,
            						shape: [B, num_queries, 4]
            batch_gt_instances,     List of gt_instances for each image.
            batch_img_metas,        List of metainfo for each image.
        #
        
        -> DETRHead.get_targets(...)
            labels_list, gt class index of each predict, `num_classes` for negative, 
                List of shape [num_queries]
            label_weights_list, 1, 
                List of shape [num_queries]
            bbox_targets_list, normed cxcywh for positive, 0 for negative, 
                List of shape [num_queries, 4]
            bbox_weights_list, 1 for positive, 0 for negative, 
                List of shape [num_queries, 4] 
            num_total_pos, number of all positive queries, 
                int
            num_total_neg, number of all negative queries, 
                int
        
        labels,        shape [B * num_queries], gt class index of each predict, `num_classes` for negative 
        label_weights, shape [B * num_queries], 1
        bbox_targets,  shape [B * num_queries, 4], normed cxcywh for positive, 0 for negative
        bbox_weights,  shape [B * num_queries, 4], 1 for positive, 0 for negative
        
        cls_scores, shape [B * num_queries, num_classes]
        
        cls_avg_factor = num_total_pos * 1.0 +  num_total_neg * self.bg_cls_weight=0
        cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        
        if isinstance(self.loss_cls, QualityFocalLoss):
        	...
		else:
			# FocalLoss
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
		
		factors, [B * num_queries, 4], image input whwh before padding to batch.
		
		bbox_preds, sigmoid cxcywh, shape: [B * num_queries, 4]
		bboxes, pred unnormed xyxy, shape: [B * num_queries, 4]
		bboxes_gt, gt unnormed xyxy for positive, 0 for negative, 
									shape: [B * num_queries, 4]
		
		# GIoULoss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
		
		# L1Loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
 		return loss_cls, loss_bbox, loss_iou
                                           
    def get_targets(...):
        # Inputs:
            cls_scores_list,        List of shape [num_queries, num_classes] for each image.
            bbox_preds_list,        List of shape [num_queries, 4] for each image.
            batch_gt_instances,     List of gt_instances for each image.
            batch_img_metas,        List of metainfo for each image.
    	#
    	
    	# Outputs:
    		labels_list, gt class index of each predict, `num_classes` for negative, 
        		List of shape [num_queries]
    		label_weights_list, 1, 
        		List of shape [num_queries]
    		bbox_targets_list, normed cxcywh for positive, 0 for negative, 
        		List of shape [num_queries, 4]
    		bbox_weights_list, 1 for positive, 0 for negative, 
        		List of shape [num_queries, 4] 
    		num_total_pos, number of all positive queries, 
    			int
    		num_total_neg, number of all negative queries, 
    			int
    	#
    	
    	
    def _get_targets_single(...):
    	# Inputs:
            cls_score,              shape [num_queries, num_classes] 
            bbox_pred, sigmoid cxcywh, 
            						shape [num_queries, 4]
            gt_instances,           InstanceData
            img_meta,               dict
        #
        
        # cls_score, logits,        shape [num_queries, num_classes] 
        # bbox_pred, unnormalized xyxy, 
        							shape [num_queries, 4]
        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        -> HungarianAssigner.assign:
        	assigned_gt_inds, shape [num_queries]
        	assigned_gt_inds=0
        	
        	assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        	
        pos_inds, matched query index, shape [num_matched]
        pos_assigned_gt_inds, matched gt index, shape [num_matched]
        pos_gt_bboxes, matched gt bbox, shape [num_matched, 4]
        
        labels, gt class index of each predict, `num_classes` for negative, shape [num_queries]
        
        label_weights, 1, shape [num_queries]
        
        bbox_targets, normed cxcywh for positive, 0 for negative, shape [num_queries, 4]
        
        bbox_weights, 1 for positive, 0 for negative, shape [num_queries, 4]
        
        # Outputs:
        	labels, gt class index of each predict, `num_classes` for negative, 
        		shape [num_queries]
        	label_weights, 1, 
        		shape [num_queries]
        	bbox_targets, normed cxcywh for positive, 0 for negative, 
        		shape [num_queries, 4]
        	bbox_weights, 1 for positive, 0 for negative, 
        		shape [num_queries, 4] 
        	pos_inds, matched query index, 
        		shape [num_matched]
        	neg_inds, no matched query index, 
        		shape [num_queries - num_matched]
        #
```



# Encoding



```
Encoder:

feat_pos: SinePositionalEncoding
reference_points: DeformableDetrTransformerEncoder.get_encoder_reference_points

Decoder:
output_proposals: DeformableDETR.gen_encoder_output_proposals

enc_outputs_coord_unact = output_proposals + reg(output_memory)

reference_points = topk(topk_coords_unact)


query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :], # first level
                num_feats=self.embed_dims // 2)
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



# PyTorch 

## Dataset

```Python
torch/utils/data/dataset.py

from typing import (
    Generic,
    ...
    TypeVar,
    ...
)

T_co = TypeVar('T_co', covariant=True)

class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    # def __getitems__(self, indices: List) -> List[T_co]:
    # Not implemented to prevent false-positives in fetcher check in
    # torch.utils.data._utils.fetch._MapDatasetFetcher

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py
```



## DataLoader



### Workflow

```
DataLoader.__iter__() -> '_BaseDataLoaderIter'
-> return DataLoader._get_iterator()
-> return _SingleProcessDataLoaderIter(DataLoader inst) if DataLoader inst.num_workers == 0
    -> _SingleProcessDataLoaderIter inst._dataset_fetcher = _DatasetKind.create_fetcher(...)
    	-> return _MapDatasetFetcher(...) if kind == _DatasetKind.Map


_SingleProcessDataLoaderIter.__next__()
-> _BaseDataLoaderIter.__next__()
    if self._sampler_iter is None:
        self._reset()
        
    data = self._next_data()
    -> _SingleProcessDataLoaderIter._next_data()
        index = self._next_index()
        -> _SingleProcessDataLoaderIter._next_index()
        	# self._sampler_iter = iter of batch_sampler if it is not None else sampler
        	return next(self._sampler_iter)
        
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        -> _MapDatasetFetcher.fetch(index)
        	def fetch(self, possibly_batched_index):
                if self.auto_collation: # batch_sampler is not None
                    if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                        data = self.dataset.__getitems__(possibly_batched_index)
                    else:
                        data = [self.dataset[idx] for idx in possibly_batched_index]
                else:
                    data = self.dataset[possibly_batched_index]
                
                return self.collate_fn(data)
        
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data
    
    return data
    
    
```



### DataLoader sampler and batch_sampler

```python
Assume dataset is not IterableDataset.

# 1.
if sampler is None and batch_sampler is None:
    if shuffle:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
            
    if batch_size is not None:
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

# 2.
if sampler is not None and batch_sampler is None:
    Must: 
        shuffle = None or False
    
    if batch_size is not None:
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

# 3. 
if sampler is None and batch_sampler is not None:
    Must: 
        batch_size = 1
        shuffle = None or False
        sampler = None
        drop_last = None or False
        
	batch_size = None
	drop_last = False
    
    sampler = SequentialSampler(dataset)

```

### DataLoader properties

```
_auto_collation = batch_sampler is not None

_index_sampler = batch_sampler or sampler

if collate_fn is None:
    if batch_sampler is not None:
        collate_fn = _utils.collate.default_collate
    else:
        collate_fn = _utils.collate.default_convert
```



### DataLoader methods

```python
torch/utils/data/dataloader.py

from typing import ..., TypeVar, Generic, ...

T_co = TypeVar('T_co', covariant=True)

class DataLoader(Generic[T_co]):
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: Optional[int]
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(...):
        ...
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   _DataPipeSerializationWrapper container makes it easier to serialize without redefining pickler
        if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            ...
            # shuffle should be False or None
            # sampler and batch_sampler should = None
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

		# If sampler is not None, then shuffle should = False or None
        # If batch_sampler is not None, then 
        # 	sampler should = None
        # 	batch_size should = 1
        # 	shuffle should = False or None
        # 	drop_last should = False or None

        if batch_sampler is not None:
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')
		
        # Set sampler if is None
        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]
		
        # Set batch_sampler if is None and batch_size is not None
        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {!r}, but got '
                             'multiprocessing_context={!r}').format(valid_start_methods, multiprocessing_context))
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError(('multiprocessing_context option should be a valid context '
                                     'object or a string specifying the start method, but got '
                                     'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError(f'{attr} attribute should not be set after {self.__class__.__name__} is initialized')

        super().__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> '_BaseDataLoaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid resetting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)

    def check_worker_number_rationality(self):
        ...
```





## _BaseDataLoaderIter

```python
class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._shared_seed = None
        self._pg = None
        if isinstance(self._dataset, IterDataPipe):
            if dist.is_available() and dist.is_initialized():
                self._pg = dist.new_group(backend="gloo")
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank
        # for other backends, pin_memory_device need to set. if not set
        # default behaviour is CUDA device. if pin_memory_device is selected
        # and pin_memory is not set, the default behaviour false.
        if (len(loader.pin_memory_device) == 0):
            self._pin_memory = loader.pin_memory and torch.cuda.is_available()
            self._pin_memory_device = None
        else:
            if not loader.pin_memory:
                warn_msg = ("pin memory device is set and pin_memory flag is not used then device pinned memory won't be used"
                            "please set pin_memory to true, if you need to use the device pin memory")
                warnings.warn(warn_msg)

            self._pin_memory = loader.pin_memory
            self._pin_memory_device = loader.pin_memory_device
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        if isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            data = self._next_data()
            self._num_yielded += 1
            if self._dataset_kind == _DatasetKind.Iterable and \
                    self._IterableDataset_len_called is not None and \
                    self._num_yielded > self._IterableDataset_len_called:
                warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                            "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                  self._num_yielded)
                if self._num_workers > 0:
                    warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                 "IterableDataset replica at each worker. Please see "
                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                warnings.warn(warn_msg)
            return data

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)

```



## _SingleProcessDataLoaderIter

```python
class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            # For BC, use default SHARDING_PRIORITIES
            torch.utils.data.graph_settings.apply_sharding(self._dataset, self._world_size, self._rank)

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data
```





## _DatasetKind

```
class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


```



## _BaseDatasetFetcher

```
class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()
```



## _MapDatasetFetcher

```
class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
```





# Issues



```
D:\proj\git\gitee_DDQ\mmdet\models\data_preprocessors\data_preprocessor.py
# Line 173, 176 should be _batch_inputs.shape[-2], _batch_inputs.shape[-1]

        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
```





```
D:\proj\git\gitee_DDQ\mmdet\models\detectors\deformable_detr.py
Line 62: should not
            # And all the prediction layers should share parameters
            # when `with_box_refine` is `True`.
```











```
D:\proj\git\mmdetection\mmdet\models\layers\transformer\utils.py
Line 93: pos_w, pos_h -> pos_h, pos_w
method: coordinate_to_encoding

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
```









```
D:\proj\git\gitee_DDQ\projects\models\ddq_detr.py
Line 343-358:

	def _init_layers(self) -> None:
        super(DDQDETRHead, self)._init_layers()
        # aux head for dense queries on encoder feature map
        self.cls_branches.append(copy.deepcopy(self.cls_branches[-1]))
        self.reg_branches.append(copy.deepcopy(self.reg_branches[-1]))

        # self.num_pred_layer is 7
        # aux head for dense queries in decoder
        self.aux_cls_branches = nn.ModuleList([
            copy.deepcopy(self.cls_branches[-1])
            for _ in range(self.num_pred_layer - 1)
        ])
        self.aux_reg_branches = nn.ModuleList([
            copy.deepcopy(self.reg_branches[-1])
            for _ in range(self.num_pred_layer - 1)
        ])
```



推理修改建议：

Line 346, 347: 添加了一个分类和回归分支头用于dense queries on encoder,是否可以改为加到351和355行的self.aux_cls_branches和 self.aux_reg_branches。因为这些都是属于对dense queries的head，改后cls_branches和aux_cls_branches 就都有7个head，分别用于queries的处理和dense queries的推理。

![image-20230804024326694](C:\Users\Phoenix\AppData\Roaming\Typora\typora-user-images\image-20230804024326694.png)







这两个问题的官方代码文件路径都是：

https://github.com/jshilong/DDQ/blob/ddq_detr/projects/models/ddq_detr.py 



训练的bug：

Line 126至128: `[tmp_dense, tmp]`变量名应互换 。

![image-20230804024600646](C:\Users\Phoenix\AppData\Roaming\Typora\typora-user-images\image-20230804024600646.png)



老师好，这是今天的会议小结：

1 对齐DDQ推理结果和训练loss，提交pr

2 询问DDQ论文作者：1 为什么要使用更新decoder self attention mask的方式筛选，而不是先直接筛选。2 Decoder的self attention mask中的distinct mask假如第i个被选中，则第i行和第i列都会设置为1，而不是只把第i行设置为1（理论上这样才会筛选第i个），原因是实验的结果是前者更优吗？

3 DDQ结束后的可选任务： 1撰写DDQ技术分享文章。 2 分析pytorch版本和paddle版本RTDETR的数据增强对齐，模型推理对齐和loss对齐。







bbox_head

|                  |      |      |      |
| ---------------- | ---- | ---- | ---- |
| reg_branches     | 1-6  | 7    | 8    |
| aux_reg_branches | 1-6  |      |      |
|                  |      |      |      |

关于bbox_head的回归branch:

reg_branches共8个。其中1-6用于dino_query+top_k_query的6层decoder回归。7用于dino_query+top_k_query的最后一层encoder回归。8用于dense_query的最后一层encoder回归。

aux_reg_branches共6个，用于dense_query的6层decoder回归。



