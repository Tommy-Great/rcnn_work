
## 项目运行方法

首先在命令行工具安装相关环境, 尤其要注意各种包的版本, 根据之前的错误, 我选择如下安装:
```bash
pip install -U openmim
mim install "mmcv>=2.0.0rc4, <2.2.0"
``` 
接下来拷贝mmdetection, 并安装3.3.0版本的mmdet:
```bash
 git clone https://github.com/open-mmlab/mmdetection.git 
 cd mmdetection
 git checkout v3.3.0
 pip install -v -e .
 cd ..
```
将服务器上的数据集解压到项目文件夹中:
```bash
 mkdir data
 tar -xzf /tmp/dataset/VOCtrainval_11-May-2012.tar.gz -C data/
```
由于mmdetection常见是按照COCO数据集的格式进行训练, 因此我们需要对VOC进行转换, 生成COCO中的annotations:
```bash
python mmdetection/tools/dataset_converters/pascal_voc.py data/VOCdevkit/ --out-dir data/VOCdevkit/ann_coco/ --out-format coco
```
接下来我们直接进行训练即可, 训练mask R-CNN:
```bash
python mmdetection/tools/train.py cfg_mask_rcnn.py
```
训练, sparse R-CNN:
```bash
python mmdetection/tools/train.py cfg_sparse_rcnn.py
```
对于cfg_mask_rcnn.py和cfg_sparse_rcnn.py是两份训练的设置文件, 
因为我们是用的COCO格式来训练VOC数据集, 
因此相对于默认的训练设置文件我们需要对data_loader和evaluator中的文件路径进行相应的更改,
下面列出一些关键的更改, 核心就是要让训练过程能够找到数据的正确位置.
```python
# model部分
bbox_head=dict(num_classes=20)
mask_head=dict(num_classes=20)
```
```python
# data_loader部分
ann_file='ann_coco/voc12_train.json'
data_prefix=dict(img='')
```
```python
# evaluator部分
data_root = 'data/VOCdevkit/'
ann_file=data_root + 'ann_coco/voc12_val.json'
```
在训练设置中, 我们还定义了结果的输出地址:
```python
work_dir = 'work_dirs/mask_rcnn'
```
也定义了训练的超参数, 包括batch size, learning rate, number of epoch等.


