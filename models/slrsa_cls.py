import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.slrsa_util import  RelationEncoding, SetAbstraction


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        self.re = RelationEncoding(radius=0.2, nsample=32, in_channel=6, mlp=[32, 32])
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=64+6, mlp=[64, 64], group_all=False)
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=20, in_channel=128+6, mlp=[128, 256], group_all=False)
        self.sa3 = SetAbstraction(npoint=1, radius=None, nsample=128, in_channel=512+6, mlp=[1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, xyz, gamma=1, hard=False):
        """
            classification task
            :param xyz: input points, [B, C ,N]
            :param hard: whether to use straight-through, Bool
            :return: prediction, [B, 40]
        """
        device = xyz.device
        B, _, _ = xyz.shape

        #   Relation Encoding Layer
        points = self.re(xyz)

        # Set Abstraction Levels
        l1_xyz, l1_points, Cos_loss1, unique_num_points1 = self.sa1(xyz, points, gamma=gamma, hard=hard)
        l2_xyz, l2_points, Cos_loss2, unique_num_points2 = self.sa2(l1_xyz, l1_points, gamma=gamma, hard=hard)
        l3_xyz, l3_points, Cos_loss3,  _ = self.sa3(l2_xyz, l2_points)

        #   FC Layers
        x = l3_points.view(B, 1024)
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2))
        x = self.fc3(x)

        #   Cosine Loss
        Cos_loss = Cos_loss1 + Cos_loss2 + Cos_loss3

        #   Unique Num of Sampled Regions
        unique_num_points1 = torch.tensor(unique_num_points1).to(device)
        unique_num_points2 = torch.tensor(unique_num_points2).to(device)
        return x, Cos_loss, unique_num_points1, unique_num_points2


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        gold = gold.long()
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

