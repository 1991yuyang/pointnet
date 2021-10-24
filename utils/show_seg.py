from __future__ import print_function
import sys
sys.path.insert(0, "..")
from show3d_balls import showpoints
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetSeg
import matplotlib.pyplot as plt
import os
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


root = r"E:\point_cloud_data\shapenetcore_partanno_segmentation_benchmark_v0"
sample_index = 0
class_choice = ["Airplane"]
npoints = 2500
rand_seed = 123
is_bias = False
#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

d = ShapeNetDataset(
    root=root,
    npoints=npoints,
    classification=False,
    class_choice=class_choice,
    split='test',
    data_augmentation=False,
    rand_seed=rand_seed
)
point, seg = d[sample_index]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy(), :]


model = PointNetSeg(seg_num_classes=d.num_seg_classes, input_dim=point.size()[0], is_bias=False)
model = nn.DataParallel(module=model, device_ids=[0])
model.load_state_dict(torch.load("../model_save/%s_seg_best.pth" % (class_choice[0],)))
model = model.cuda(0)
model.eval()

point = point.unsqueeze(0).cuda(0)

with torch.no_grad():
    pred, _ = model(point)
pred_choice = pred.detach().cpu().transpose(1, 2).data.max(2)[1]

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
showpoints(np.transpose(point_np, axes=[1, 0]), gt, pred_color)
