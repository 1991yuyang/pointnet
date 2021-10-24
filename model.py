import torch as t
from torch import nn


class TNet(nn.Module):

    def __init__(self, input_dim, is_bias):
        """
        generate affine matrix
        :param input_dim: point cloud data dimension, for example input_dim of (x, y, z) is 3
        :param is_bias: bool type, "True" include bias in conv and fc layer, otherwise "False"
        """
        super(TNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512, bias=is_bias),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=is_bias),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=256, out_features=input_dim * input_dim)
        self.input_dim = input_dim

    def forward(self, x):
        """

        :param x: shape like (N, input_dim, n), N is batch size, input_dim is dimension of every point, n is the count of input points
        :return:
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = t.max(x, dim=2).values
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view((x.size()[0], self.input_dim, self.input_dim))
        iden = t.eye(self.input_dim, self.input_dim).repeat(x.size()[0], 1, 1).cuda(0)
        x = x + iden
        return x


class ClsHead(nn.Module):

    def __init__(self, num_classes, is_bias):
        super(ClsHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=is_bias)
        )

    def forward(self, global_feature):
        x = self.conv(global_feature)
        x = x.view((x.size()[0], -1))
        return x


class SegHead(nn.Module):

    def __init__(self, seg_num_classes, is_bias):
        super(SegHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=seg_num_classes, kernel_size=1, stride=1, padding=0, bias=is_bias)
        )

    def forward(self, local_feature, global_feature):
        concate_feature = t.cat((local_feature, global_feature.repeat(1, 1, local_feature.size()[-1])), dim=1)
        x = self.conv1(concate_feature)
        x = self.conv2(x) # (N, seg_num_classes, n)
        return x


class LocalFeat(nn.Module):

    def __init__(self, input_dim, is_bias):
        """
        extract local feature
        :param input_dim: point cloud data dimension, for example input_dim of (x, y, z) is 3
        :param is_bias: bool type, "True" include bias in conv and fc layer, otherwise "False"
        """
        super(LocalFeat, self).__init__()
        self.tnet_trans_input = TNet(input_dim, is_bias)
        self.tnet_trans_feat = TNet(64, is_bias)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )

    def forward(self, x):
        """

        :param x: shape like (N, input_dim, n), N is batch size, input_dim is dimension of every point, n is the count of input points
        :return:
        """
        input_trans_matrix = self.tnet_trans_input(x)
        x = t.bmm(x.transpose(2, 1).contiguous(), input_trans_matrix).transpose(2, 1).contiguous()
        x = self.conv1(x)
        feat_trans_matrix = self.tnet_trans_feat(x)
        x = t.bmm(x.transpose(2, 1).contiguous(), feat_trans_matrix).transpose(2, 1).contiguous()  # (N, 64, n)
        return x, feat_trans_matrix


class GlobalFeat(nn.Module):

    def __init__(self, is_bias):
        super(GlobalFeat, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=is_bias),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        """

        :param x: local feature extracted from PointNetFeat
        :return:
        """
        x = self.conv1(x)
        global_feature = self.pool(x)  # (N, 1024, 1)
        return global_feature


class PointNetCls(nn.Module):

    def __init__(self, num_classes, is_bias, input_dim):
        super(PointNetCls, self).__init__()
        self.local_feat_extractor = LocalFeat(input_dim, is_bias)
        self.global_feat_extractor = GlobalFeat(is_bias)
        self.cls_head = ClsHead(num_classes, is_bias)

    def forward(self, x):
        local_feat, feat_trans_matrix = self.local_feat_extractor(x)
        global_feat = self.global_feat_extractor(local_feat)
        cls_output = self.cls_head(global_feat)
        return cls_output, feat_trans_matrix


class PointNetSeg(nn.Module):

    def __init__(self, seg_num_classes, input_dim, is_bias):
        super(PointNetSeg, self).__init__()
        self.local_feat_extractor = LocalFeat(input_dim, is_bias)
        self.global_feat_extractor = GlobalFeat(is_bias)
        self.seg_head = SegHead(seg_num_classes, is_bias)

    def forward(self, x):
        local_feature, feat_trans_matrix = self.local_feat_extractor(x)
        global_feature = self.global_feat_extractor(local_feature)
        seg_output = self.seg_head(local_feature, global_feature)  # (N, seg_num_classes, n)
        return seg_output, feat_trans_matrix


def regularization_item(trans_matrix):
    I = t.eye(trans_matrix.size()[-1]).unsqueeze(0).cuda(0)
    loss = t.mean(t.norm(t.bmm(trans_matrix, trans_matrix.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == "__main__":
    tnet = PointNetSeg(seg_num_classes=10, is_bias=False, input_dim=3).cuda(0)
    d = t.randn(2, 3, 2500).cuda(0)
    output, trans_matrix = tnet(d)
    print(regularization_item(trans_matrix))