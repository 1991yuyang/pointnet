# pointnet
## Dataset
Download dataset from https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
## Train  
Before training classification or segmentation tasks, you first need to start visdom with the following command:  
```
python -m visdom.server
```
### Classification 
First, you need to configure the following items in train_cls.py  
```
root = r"./shapenetcore_partanno_segmentation_benchmark_v0"
epoch = 100
train_batch_size = 16
test_batch_size = 64
lr = 0.001
lr_de_rate = 0.5
lr_de_epoch = [20, 30, 50, 70]
print_loss_accu_every_step = 100
npoints = 2500
num_workers = 4
only_test = True
rand_seed = 123
```
### Segmentation  
First, you need to configure the following items in train_seg.py
## Show Segmentation Result  
## Segmentation Results  
<p float="left">
  <img src="test_seg_imgs/guitar1.png" width="200" height="200"/>
  <img src="test_seg_imgs/guitar2.png" width="200" height="200"/>
  <img src="test_seg_imgs/guitar3.png" width="200" height="200"/>
  <img src="test_seg_imgs/guitar4.png" width="200" height="200"/>
</p>
<p float="left">
  <img src="test_seg_imgs/airplane1.png" width="200" height="200"/>
  <img src="test_seg_imgs/airplane2.png" width="200" height="200"/>
  <img src="test_seg_imgs/airplane3.png" width="200" height="200"/>
  <img src="test_seg_imgs/airplane4.png" width="200" height="200"/>
</p>

