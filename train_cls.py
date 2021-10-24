import torch as t
from torch import nn, optim
import os
from dataset import make_loader, ShapeNetDataset
from model import PointNetCls, regularization_item
from torch.utils import data
from tqdm import tqdm
from visdom import Visdom
import random
random.seed(1000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


root = r"E:\point_cloud_data\shapenetcore_partanno_segmentation_benchmark_v0" # download dataset from https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip

epoch = 100
train_batch_size = 16
test_batch_size = 64
lr = 0.001
lr_de_rate = 0.5
lr_de_epoch = [20, 30, 50, 70]
print_loss_accu_every_step = 100
softmax_op = nn.Softmax(dim=1)
input_dim = 3
is_bias = False
npoints = 2500
num_workers = 4
only_test = True
rand_seed = 123
ShapeNetDataset(root, npoints=npoints, classification=True, class_choice=None, split='train', data_augmentation=True, rand_seed=rand_seed)  # generate classification_info.json and segmentation_info.json
with open("classification_info.json", "r", encoding="utf-8") as file:
    classification_info = eval(file.read())
num_classes = len(classification_info)
print(classification_info)
if not only_test:
    wind_train_loss = Visdom()
    wind_train_accu = Visdom()
    wind_valid_loss = Visdom()
    wind_valid_accu = Visdom()
    train_total_step = 0
    valid_total_step = 0


def calc_accu(model_output, target, softmax_op):
    accu = (t.argmax(softmax_op(model_output), dim=1) == target).sum().item() / model_output.size()[0]
    return accu


def train_epoch(model, current_epoch, all_epoch, train_loader, criterion, optimizer, softmax_op):
    global train_total_step
    model.train()
    all_steps = len(train_loader)
    current_step = 1
    for d_train, l_train in train_loader:
        d_train_cuda = d_train.cuda(0)
        l_train_cuda = l_train.squeeze().cuda(0)
        train_output, feat_trans_matrix = model(d_train_cuda)
        train_loss = criterion(train_output, l_train_cuda) + regularization_item(feat_trans_matrix) * 0.001
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_accu = calc_accu(train_output, l_train_cuda, softmax_op)
        if current_step % print_loss_accu_every_step == 0:
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, train_accu:%.5f" % (current_epoch, all_epoch, current_step, all_steps, train_loss.item(), train_accu))
        if (train_total_step + 1) % print_loss_accu_every_step == 0:
            wind_train_loss.line([train_loss.detach().cpu()], [train_total_step], win='train_loss', update='append', opts=dict(title='train_loss'))
            wind_train_accu.line([train_accu], [train_total_step], win='train_accu', update='append', opts=dict(title='train_accu'))
        train_total_step += 1
        current_step += 1
    return model


def valid_epoch(model, current_epoch, all_epoch, valid_loader, criterion, softmax_op):
    global valid_total_step
    model.eval()
    cum_loss = 0
    cum_accu = 0
    all_steps = len(valid_loader)
    bar = tqdm(valid_loader)
    bar.set_description("validation")
    for d_valid, l_valid in bar:
        d_valid_cuda = d_valid.cuda(0)
        l_valid_cuda = l_valid.squeeze().cuda(0)
        with t.no_grad():
            valid_output, feat_trans_matrix = model(d_valid_cuda)
            valid_loss = criterion(valid_output, l_valid_cuda) + 0.001 * regularization_item(feat_trans_matrix)
            valid_accu = calc_accu(valid_output, l_valid_cuda, softmax_op)
            cum_loss += valid_loss.item()
            cum_accu += valid_accu
        if (valid_total_step + 1) % print_loss_accu_every_step == 0:
            wind_valid_loss.line([valid_loss.detach().cpu()], [valid_total_step], win='valid_loss', update='append', opts=dict(title='valid_loss'))
            wind_valid_accu.line([valid_accu], [valid_total_step], win='valid_accu', update='append', opts=dict(title='valid_accu'))
        valid_total_step += 1
    avg_loss = cum_loss / all_steps
    avg_accu = cum_accu / all_steps
    print("#################valid epoch:%d/%d result#################" % (current_epoch, all_epoch))
    print("valid_loss:%.5f, valid_accu:%.5f" % (avg_loss, avg_accu))
    return model, avg_loss


def test(softmax_op):
    model = PointNetCls(num_classes=num_classes, input_dim=input_dim, is_bias=is_bias)
    model = nn.DataParallel(module=model, device_ids=[0])
    model.load_state_dict(t.load("model_save/cls_best.pth"))
    model.cuda(0)
    model.eval()
    total_correct_count = 0
    total_test_sample_count = 0
    test_set = ShapeNetDataset(root, npoints=npoints, classification=True, class_choice=None, split='test', data_augmentation=False, rand_seed=rand_seed)
    test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, drop_last=False)
    bar = tqdm(test_loader)
    bar.set_description("testing")
    for d_test, l_test in bar:
        d_test_cuda = d_test.cuda(0)
        l_test_cuda = l_test.squeeze().cuda(0)
        with t.no_grad():
            test_output, _ = model(d_test_cuda)
            total_correct_count += (t.argmax(softmax_op(test_output), dim=1) == l_test_cuda).sum().item()
            total_test_sample_count += test_output.size()[0]
    test_set_accu = total_correct_count / total_test_sample_count
    print("######################test result##############################")
    print("test set accuracy: %.5f" % (test_set_accu,))


def main():
    if not only_test:
        best_valid_loss = float("inf")
        model = PointNetCls(num_classes=num_classes, input_dim=input_dim, is_bias=is_bias)
        model = nn.DataParallel(module=model, device_ids=[0])
        model = model.cuda(0)
        criterion = nn.CrossEntropyLoss().cuda(0)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_de_epoch, lr_de_rate)
        for e in range(epoch):
            train_loader = make_loader(root, classification=True, split="train", data_augmentation=True, npoints=npoints, batch_size=train_batch_size, class_choice=None, num_workers=num_workers, rand_seed=rand_seed)
            valid_loader = make_loader(root, classification=True, split="val", data_augmentation=False, npoints=npoints, batch_size=train_batch_size, class_choice=None, num_workers=num_workers, rand_seed=rand_seed)
            model = train_epoch(model, e + 1, epoch, train_loader, criterion, optimizer, softmax_op)
            model, valid_loss = valid_epoch(model, e + 1, epoch, valid_loader, criterion, softmax_op)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print("saving best model.......")
                t.save(model.state_dict(), "model_save/cls_best.pth")
            print("saving epoch model.......")
            t.save(model.state_dict(), "model_save/cls_epoch.pth")
            lr_sched.step()
    if os.path.exists("model_save/cls_best.pth"):
        test(softmax_op)
    else:
        raise(Exception("there is not cls_best.pth in model_save"))


if __name__ == "__main__":
    main()
