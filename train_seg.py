import torch as t
from torch import nn, optim
from model import PointNetSeg, regularization_item
from dataset import make_loader, ShapeNetDataset
from torch.utils import data
from tqdm import tqdm
import os
from visdom import Visdom
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


root = r"E:\point_cloud_data\shapenetcore_partanno_segmentation_benchmark_v0" # download dataset from https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
epoch = 200
train_batch_size = 4
test_batch_size = 32
lr = 0.01
lr_de_rate = 0.5
lr_de_epoch = [20, 40, 60, 80, 100, 120, 140, 160]
print_loss_iou_every_step = 10
softmax_op = nn.Softmax(dim=2)
input_dim = 3
is_bias = False
npoints = 2500
num_workers = 4
only_test = False
rand_seed = 123
class_choice = ["Guitar"]  # segmentation task need specify one class, should be a list, for example ["class1"]
s_ = ShapeNetDataset(root, npoints=npoints, classification=False, class_choice=class_choice, split='train', data_augmentation=True, rand_seed=rand_seed)  # generate classification_info.json and segmentation_info.json
num_seg_classes = s_.num_seg_classes
print("class name:", class_choice[0])
print("class number of %s segmentation:%d" % (class_choice[0], num_seg_classes))
if not only_test:
    wind_train_loss = Visdom()
    wind_train_iou = Visdom()
    wind_valid_loss = Visdom()
    wind_valid_iou = Visdom()
    train_total_step = 0
    valid_total_step = 0


def calc_mean_iou(output, target, softmax_op):
    pred_choice = t.argmax(softmax_op(output.transpose(1, 2)), dim=2)
    shape_ious = []
    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(target.shape[0]):
        parts = range(num_seg_classes)  # np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    batch_mean_iou = np.mean(shape_ious)
    return batch_mean_iou, shape_ious


def train_epoch(criterion, current_epoch, model, train_loader, softmax_op, optimizer):
    global train_total_step
    model.train()
    all_steps = len(train_loader)
    current_step = 1
    for d_train, l_train in train_loader:
        d_train_cuda = d_train.cuda(0)
        l_train_cuda = l_train.cuda(0)
        train_output, feat_trans_matrix = model(d_train_cuda)
        train_loss = criterion(train_output, l_train_cuda) + 0.001 * regularization_item(feat_trans_matrix)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_mean_iou, _ = calc_mean_iou(train_output, l_train_cuda, softmax_op)
        if current_step % print_loss_iou_every_step == 0:
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, train_iou:%.5f" % (current_epoch, epoch, current_step, all_steps, train_loss.item(), train_mean_iou))
        current_step += 1
        if (train_total_step + 1) % print_loss_iou_every_step == 0:
            wind_train_loss.line([train_loss.detach().cpu()], [train_total_step], win='%s_train_loss' % (class_choice[0],), update='append', opts=dict(title='train_loss'))
            wind_train_iou.line([train_mean_iou], [train_total_step], win='%s_train_iou' % (class_choice[0],), update='append', opts=dict(title='train_iou'))
        train_total_step += 1
    return model


def valid_epoch(model, valid_loader, criterion, softmax_op, current_epoch):
    global valid_total_step
    model.eval()
    all_steps = len(valid_loader)
    cum_loss = 0
    cum_iou = 0
    bar = tqdm(valid_loader)
    bar.set_description("validation")
    for d_valid, l_valid in bar:
        d_valid_cuda = d_valid.cuda(0)
        l_valid_cuda = l_valid.cuda(0)
        with t.no_grad():
            valid_output, feat_trans_matrix = model(d_valid_cuda)
            valid_loss = criterion(valid_output, l_valid_cuda) + 0.001 * regularization_item(feat_trans_matrix)
            valid_mean_iou, _ = calc_mean_iou(valid_output, l_valid_cuda, softmax_op)
            cum_loss += valid_loss.item()
            cum_iou += valid_mean_iou
        if (valid_total_step + 1) % print_loss_iou_every_step == 0:
            wind_valid_loss.line([valid_loss.detach().cpu()], [valid_total_step], win='%s_valid_loss' % (class_choice[0],), update='append', opts=dict(title='valid_loss'))
            wind_valid_iou.line([valid_mean_iou], [valid_total_step], win='%s_valid_iou' % (class_choice[0],), update='append', opts=dict(title='valid_iou'))
        valid_total_step += 1
    if all_steps != 0:
        valid_loss = cum_loss / all_steps
        valid_iou = cum_iou / all_steps
    else:
        valid_loss = cum_loss
        valid_iou = cum_iou
    print("#################valid epoch:%d/%d result#################" % (current_epoch, epoch))
    print("valid_loss:%.5f, valid_iou:%.5f" % (valid_loss, valid_iou))
    return model, valid_loss


def test(softmax_op):
    model = PointNetSeg(seg_num_classes=num_seg_classes, is_bias=is_bias, input_dim=input_dim)
    model = nn.DataParallel(module=model, device_ids=[0])
    model.load_state_dict(t.load("model_save/%s_seg_best.pth" % (class_choice[0],)))
    model = model.cuda(0)
    model.eval()
    test_set = ShapeNetDataset(root, npoints=npoints, classification=False, class_choice=class_choice, split='test', data_augmentation=False, rand_seed=rand_seed)
    test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, drop_last=False)
    bar = tqdm(test_loader)
    bar.set_description("testing")
    total_test_sample_count = 0
    test_shape_ious = []
    for d_test, l_test in bar:
        d_test_cuda = d_test.cuda(0)
        l_test_cuda = l_test.cuda(0)
        total_test_sample_count += d_test_cuda.size()[0]
        with t.no_grad():
            test_output, _ = model(d_test_cuda)
            _, shape_ious = calc_mean_iou(test_output, l_test_cuda, softmax_op)
            test_shape_ious.extend(shape_ious)
    test_avg_iou = np.mean(test_shape_ious)
    print("######################test result##############################")
    print("test set iou: %.5f" % (test_avg_iou,))


def main():
    if not only_test:
        best_valid_loss = float("inf")
        model = PointNetSeg(seg_num_classes=num_seg_classes, input_dim=input_dim, is_bias=is_bias)
        model = nn.DataParallel(module=model, device_ids=[0])
        model = model.cuda(0)
        criterion = nn.CrossEntropyLoss().cuda(0)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_de_epoch, lr_de_rate)
        for e in range(epoch):
            train_loader = make_loader(root, classification=False, split="train", data_augmentation=True, npoints=npoints, batch_size=train_batch_size, class_choice=class_choice, num_workers=num_workers, rand_seed=rand_seed)
            valid_loader = make_loader(root, classification=False, split="val", data_augmentation=False, npoints=npoints, batch_size=train_batch_size, class_choice=class_choice, num_workers=num_workers, rand_seed=rand_seed)
            model = train_epoch(criterion, e + 1, model, train_loader, softmax_op, optimizer)
            model, valid_loss = valid_epoch(model, valid_loader, criterion, softmax_op, e + 1)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print("saving best model.......")
                t.save(model.state_dict(), "model_save/%s_seg_best.pth" % (class_choice[0],))
            print("saving epoch model.......")
            t.save(model.state_dict(), "model_save/%s_seg_epoch.pth" % (class_choice[0],))
            lr_sched.step()
    if os.path.exists("model_save/%s_seg_best.pth" % (class_choice[0],)):
        test(softmax_op)
    else:
        raise(Exception("there is not %s_seg_best.pth in model_save" % (class_choice[0],)))


if __name__ == "__main__":
    main()