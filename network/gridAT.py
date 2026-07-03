
import torch
import torch.nn as nn

from mlp import MLP
from grid_pooling_func import AvgPoolingModule


def get_conv_activation(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    else:
        raise NotImplementedError('Not supported activation')


def define_convs(conv_dim, num_conv, activation, latent_size, kernel_size, padding=0):
    if num_conv == 0:
        return None

    conv = getattr(nn, f'Conv{conv_dim}d')
    seq = []
    ls = latent_size
    for i in range(num_conv):
        seq.append(conv(ls, ls, kernel_size=kernel_size, padding=padding))
        seq.append(get_conv_activation(activation))

    return nn.Sequential(*seq)


# 新增注意力聚合模块
class AttentionAggregation(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 注意力权重计算的MLP
        self.query_proj = nn.Linear(feat_dim, feat_dim)  # 中心特征投影为查询
        self.key_proj = nn.Linear(feat_dim, feat_dim)    # 近邻特征投影为键
        self.value_proj = nn.Linear(feat_dim, feat_dim)  # 近邻特征投影为值
        self.softmax = nn.Softmax(dim=1)  # 在近邻维度做归一化

    def forward(self, feat):
        # feat: (N_pts, N_knn, feat_dim) - 输入特征
        N_pts, N_knn, feat_dim = feat.shape
        
        # 以每个KNN组的第一个点作为中心参考点
        center_feat = feat[:, 0:1, :]  # (N_pts, 1, feat_dim)
        
        # 计算注意力权重：Q*K^T / sqrt(dim)
        query = self.query_proj(center_feat)  # (N_pts, 1, feat_dim)
        key = self.key_proj(feat)             # (N_pts, N_knn, feat_dim)
        attn_scores = torch.bmm(query, key.transpose(1, 2)) / (feat_dim ** 0.5)  # (N_pts, 1, N_knn)
        attn_weights = self.softmax(attn_scores)  # 归一化权重
        
        # 加权聚合近邻特征：注意力权重 * 价值特征
        value = self.value_proj(feat)  # (N_pts, N_knn, feat_dim)
        agg_feat = torch.bmm(attn_weights, value)  # (N_pts, 1, feat_dim)
        
        return agg_feat


class PointGridEncoder(nn.Module):
    def __init__(self, params):
        super(PointGridEncoder, self).__init__()
        p = params
        self.grid_size = p.grid_size
        self.mlp_feat = MLP(** p.mlp)
        self.avg_pooling = AvgPoolingModule(self.grid_size)

        # 获取MLP输出的特征维度（用于初始化注意力模块）
        self.feat_dim = p.mlp['size'][-1]
        
        if 'max_pooling' not in p:
            self.use_max_pooling = False
            # 替换原有1D卷积为注意力聚合
            self.attention_agg = AttentionAggregation(self.feat_dim)
        else:
            self.use_max_pooling = True

        self.grid_feat = define_convs(**p.grid_conv)


    def forward(self, model_input):
        # pos: (N_pts, N_knn, 3)
        pc_KNN_pos = model_input['pc_KNN_pos']
        points = model_input['info']['points']
        feat = self.mlp_feat.forward_simple(pc_KNN_pos)
        # After MLP  feat: (N_pts, N_knn, feat_dim)
        
        if self.use_max_pooling:
            feat = torch.max(feat, dim=1, keepdim=True)[0]
        else:
            # 使用注意力机制聚合特征（替代原1D卷积）
            feat = self.attention_agg(feat)  # 输出: (N_pts, 1, feat_dim)
        
        # After aggregation  feat: (N_pts, 1, feat_dim)
        Nf = feat.shape[-1]
        Np = pc_KNN_pos.shape[0]
        feat = feat.view((Np, Nf))
        
        temp_grid = self.avg_pooling(feat, points)
        if self.grid_feat is not None:
            temp_grid = temp_grid.permute((3,0,1,2))
            # (N_gfeat, k,k,k)
            feature_grid = self.grid_feat(temp_grid).permute((1,2,3,0))
            # final feature: (k,k,k, N_gfeat）
        else:
            feature_grid = temp_grid

        return feature_grid


if __name__ == '__main__':
    from dotted.collection import DottedDict
    params = DottedDict({
        'max_pooling': False,
        'grid_size': 8,
        'mlp': {
            'size': [3,128,128],
            'activation_type': 'lrelu',
            'num_pos_encoding': -1
        },
        'grid_conv': {
            'latent_size': 128,
            'conv_dim': 3,
            'num_conv': 3,
            'activation': 'lrelu',
            'kernel_size': 3,
            'padding': 1,
        }
    })

    points = torch.rand(10, 3)*2 - 1
    model_input = {
        'pc_KNN_pos': torch.rand((10, 4, 3)),
        'points': points
    }

    encoder = PointGridEncoder(params)
    feat = encoder.forward(model_input)
    print(feat.shape)