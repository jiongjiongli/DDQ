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
# mmengine==0.6.0

# Install pytorch.
# Option 1: install directly.
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Option 2: Download and then install.
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl

pip install ~/data/torch-1.9.0+cu111-cp38-cp38-linux_x86_64.whl
pip install ~/data/torchvision-0.10.0+cu111-cp38-cp38-linux_x86_64.whl

pip install mmengine==0.8.4
pip install mmcv

cd ~/ddq/new/mmdetection/
pip install -v -e .
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

### 

```
cd ~/ddq/legacy/proj/DDQ
chmod +x ./repro/gpu_4_train.sh
dsub -s ./repro/gpu_4_train.sh
```



### 4.1.2 DDQ DETR

```
cd ~/ddq/legacy/proj/ddq_detr
cp ./repro/train.sh ./repro/launch_train.sh
chmod +x ./repro/launch_train.sh
dsub -s ./repro/launch_train.sh
```





```
python tools/train.py projects/configs/ddq_detr/ddq-detr-4scale_r50_8xb2-12e_coco.py ./exp/ddq_detr
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



```
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
generate_dn_bbox_query

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

