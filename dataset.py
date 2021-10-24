from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
# download dataset from https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip


def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 rand_seed=123):
        """

        :param root: shapenet dataset root dir
        :param npoints: sample count of every point cloud
        :param classification: True indicate classification task, False indicate segmentation task
        :param class_choice: choice some class to train or valid
        :param split: "train" indicate load train set, "val" indicate validation set
        :param data_augmentation: if True, use data aug, otherwise not use
        :param rand_seed: int type, used when random sample from a point set
        """
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        classification_info = json.dumps(self.classes)
        if not os.path.exists("classification_info.json"):
            with open("classification_info.json", "w", encoding="utf-8") as file:
                file.write(classification_info)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        if not os.path.exists("segmentation_info.json"):
            with open("segmentation_info.json", "w", encoding="utf-8") as file:
                file.write(json.dumps(self.seg_classes))
        self.rand_seed = rand_seed

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # print(point_set.shape, seg.shape)
        np.random.seed(self.rand_seed)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set.transpose(0, 1), cls
        else:
            return point_set.transpose(0, 1), seg - 1

    def __len__(self):
        return len(self.datapath)


def make_loader(root, classification, split, data_augmentation, npoints, class_choice, batch_size, num_workers, rand_seed):
    """

    :param root: shapenet dataset root dir
    :param npoints: sample count of every point cloud
    :param classification: True indicate classification task, False indicate segmentation task
    :param class_choice: choice some class to train or valid
    :param split: "train" indicate load train set, "val" indicate validation set
    :param data_augmentation: if True, use data aug, otherwise not use
    :param batch_size: batch_size
    :param num_workers: num_workers
    :param rand_seed: int type, used when random sample from a point set
    """
    s = ShapeNetDataset(root,
                 npoints,
                 classification,
                 class_choice,
                 split,
                 data_augmentation,
                 rand_seed)
    loader = data.DataLoader(s, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return loader


if __name__ == '__main__':
    loader = make_loader(root=r"E:\point_cloud_data\shapenetcore_partanno_segmentation_benchmark_v0", classification=False, split="val", data_augmentation=False, npoints=2500, batch_size=8, class_choice=["Motorbike"], num_workers=0, rand_seed=123)
    for d, l in loader:
        print(d.size())
        print(l.size())
    # get_segmentation_classes(datapath)
