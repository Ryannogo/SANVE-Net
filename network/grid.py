import torch
import torch.nn as nn
import torch.nn.functional as F

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


# -------------------------- 新增：DGCNN核心模块 --------------------------
class DynamicAdjacency(nn.Module):
    """基于特征相似度动态构建邻接关系（替代固定KNN）"""
    def __init__(self, dynamic_k):
        super().__init__()
        self.dynamic_k = dynamic_k  # 动态选择的近邻数（可与原KNN一致，如8）

    def forward(self, feat):
        # feat: (N_pts, feat_dim) - 每个点的全局特征（MLP输出后展平）
        N_pts = feat.shape[0]
        
        # 1. 计算特征相似度矩阵（余弦相似度，范围[-1,1]，值越大越相似）
        feat_normalized = F.normalize(feat, dim=1)  # 特征归一化（避免尺度影响）
        sim_matrix = torch.matmul(feat_normalized, feat_normalized.T)  # (N_pts, N_pts)
        
        # 2. 动态选择Top-K近邻（排除自身：对角线设为最小值）
        sim_matrix.fill_diagonal_(-float('inf'))  # 自身不参与近邻选择
        topk_sim, topk_idx = torch.topk(sim_matrix, k=self.dynamic_k, dim=1)  # (N_pts, K), (N_pts, K)
        
        # 3. 提取动态近邻的特征（按索引筛选）
        # 构建索引：(N_pts, K) -> 扩展为(N_pts*K)，便于索引
        batch_idx = torch.arange(N_pts).unsqueeze(1).repeat(1, self.dynamic_k).flatten().to(feat.device)
        dynamic_neighbor_feat = feat[batch_idx, :][topk_idx.flatten(), :].view(N_pts, self.dynamic_k, -1)
        
        return dynamic_neighbor_feat, topk_idx  # 动态近邻特征 + 近邻索引


class EdgeConv(nn.Module):
    """DGCNN核心的EdgeConv模块：对动态近邻进行图卷积聚合"""
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        # EdgeConv逻辑：对“中心特征-近邻特征”的差值+近邻特征做卷积
        self.conv = nn.Sequential(
            nn.Conv2d(2 * feat_dim, hidden_dim, kernel_size=1),  # 1x1卷积（无空间卷积，仅特征融合）
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, feat_dim, kernel_size=1)
        )

    def forward(self, center_feat, neighbor_feat):
        # center_feat: (N_pts, 1, feat_dim) - 中心点点特征
        # neighbor_feat: (N_pts, K, feat_dim) - 动态近邻特征
        
        # 1. 构建Edge特征：中心特征扩展后，与近邻特征拼接（差值+近邻）
        center_feat_expand = center_feat.repeat(1, neighbor_feat.shape[1], 1)  # (N_pts, K, feat_dim)
        edge_feat = torch.cat([center_feat_expand - neighbor_feat, neighbor_feat], dim=-1)  # (N_pts, K, 2*feat_dim)
        
        # 2. 适配卷积输入格式：(N_pts, 2*feat_dim, K, 1)（batch, channel, height, width）
        edge_feat = edge_feat.permute(0, 2, 1).unsqueeze(-1)  # (N_pts, 2*feat_dim, K, 1)
        
        # 3. 图卷积聚合：对每个中心的K个近邻特征做卷积，再取最大值（保留显著特征）
        conv_feat = self.conv(edge_feat)  # (N_pts, feat_dim, K, 1)
        agg_feat = torch.max(conv_feat, dim=2, keepdim=False).values  # (N_pts, feat_dim, 1)
        
        # 4. 恢复维度：(N_pts, 1, feat_dim)（与原聚合输出格式一致）
        return agg_feat.permute(0, 2, 1)


# -------------------------- 原有PointGridEncoder修改 --------------------------
class PointGridEncoder(nn.Module):
    def __init__(self, params):
        super(PointGridEncoder, self).__init__()
        p = params
        self.grid_size = p.grid_size
        self.mlp_feat = MLP(** p.mlp)
        self.avg_pooling = AvgPoolingModule(self.grid_size)

        # 1. 获取关键参数：MLP输出特征维度、动态近邻数K
        self.feat_dim = p.mlp['size'][-1]
        self.dynamic_k = p.get('dynamic_k', 8)  # 动态近邻数（默认8，与原KNN一致）
        self.hidden_dim = p.get('edgeconv_hidden', 128)  # EdgeConv隐藏层维度

        # 2. 初始化动态图卷积模块（替代原1D卷积/注意力）
        self.dynamic_adj = DynamicAdjacency(dynamic_k=self.dynamic_k)
        self.edge_conv = EdgeConv(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim)

        # 3. 保留原max_pooling分支（可选切换）
        self.use_max_pooling = p.get('max_pooling', False)

        self.grid_feat = define_convs(**p.grid_conv)


    def forward(self, model_input):
        # 原输入：固定KNN坐标（pc_KNN_pos）、点云坐标（points）
        pc_KNN_pos = model_input['pc_KNN_pos']  # (N_pts, fixed_K, 3) - 仅用于MLP初始特征提取
        points = model_input['info']['points']  # (N_pts, 3) - 点云全局坐标

        # Step 1: MLP提取初始特征（基于固定KNN坐标，为后续动态邻接计算做准备）
        # 注意：此处先用固定KNN坐标提特征，后续会用特征相似度重构邻接
        init_feat = self.mlp_feat.forward_simple(pc_KNN_pos)  # (N_pts, fixed_K, feat_dim)
        # 展平为每个点的全局特征：取固定KNN的平均（或最大值），用于动态邻接计算
        center_feat = torch.mean(init_feat, dim=1, keepdim=True)  # (N_pts, 1, feat_dim)
        center_feat_flat = center_feat.squeeze(1)  # (N_pts, feat_dim) - 用于动态近邻筛选

        # Step 2: 动态图卷积聚合（核心修改）
        if self.use_max_pooling:
            # 保留原max_pooling分支
            agg_feat = torch.max(init_feat, dim=1, keepdim=True)[0]  # (N_pts, 1, feat_dim)
        else:
            # 2.1 基于特征相似度动态构建邻接（替换固定KNN）
            dynamic_neighbor_feat, _ = self.dynamic_adj(center_feat_flat)  # (N_pts, K, feat_dim)
            # 2.2 EdgeConv图卷积聚合
            agg_feat = self.edge_conv(center_feat, dynamic_neighbor_feat)  # (N_pts, 1, feat_dim)

        # Step 3: 后续流程与原代码完全一致（无需修改）
        Nf = agg_feat.shape[-1]
        Np = pc_KNN_pos.shape[0]
        agg_feat_flat = agg_feat.view((Np, Nf))  # (N_pts, feat_dim)

        # 网格池化
        temp_grid = self.avg_pooling(agg_feat_flat, points)
        # 3D卷积（若有）
        if self.grid_feat is not None:
            temp_grid = temp_grid.permute((3, 0, 1, 2))  # (feat_dim, grid_size, grid_size, grid_size)
            feature_grid = self.grid_feat(temp_grid).permute((1, 2, 3, 0))  # (grid_size, grid_size, grid_size, feat_dim)
        else:
            feature_grid = temp_grid

        return feature_grid


# -------------------------- 测试代码 --------------------------
if __name__ == '__main__':
    from dotted.collection import DottedDict
    # 配置参数（新增dynamic_k和edgeconv_hidden）
    params = DottedDict({
        'max_pooling': False,  # 启用动态图卷积分支
        'grid_size': 8,
        'dynamic_k': 8,  # 动态近邻数（可与原KNN一致）
        'edgeconv_hidden': 128,  # EdgeConv隐藏层维度
        'mlp': {
            'size': [3, 128, 128],  # MLP输入3（坐标），输出128（feat_dim）
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

    # 模拟输入
    N_pts = 100  # 点云数量
    fixed_K = 8  # 原输入的固定KNN数
    points = torch.rand(N_pts, 3) * 2 - 1  # 点云全局坐标
    model_input = {
        'pc_KNN_pos': torch.rand((N_pts, fixed_K, 3)),  # 原固定KNN坐标
        'info': {'points': points}
    }

    # 初始化并测试
    encoder = PointGridEncoder(params)
    feature_grid = encoder(model_input)
    print(f"输出网格特征形状: {feature_grid.shape}")  # 应输出: torch.Size([8, 8, 8, 128])