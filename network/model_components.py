import torch
from torch import nn


def scaled_dot_product_attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    attention_weight = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', attention_weight, value), attention_weight


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, num_total_attention_feat: int):
        super(MultiHeadedAttention, self).__init__()

        assert num_total_attention_feat % num_heads == 0, ValueError(
            f"num_total_attention_feat ({num_total_attention_feat}) % num_heads ({num_heads}) is not 0 ({num_total_attention_feat % num_heads})")
        self.dim = num_total_attention_feat // num_heads
        self.num_heads = num_heads

        self.proj = nn.ModuleList(
            [nn.Conv1d(num_total_attention_feat, num_total_attention_feat, kernel_size=1) for _ in range(3)])

        self.merge = nn.Conv1d(num_total_attention_feat, num_total_attention_feat, kernel_size=1)

        self.post_merge = nn.Sequential(
            nn.Conv1d(2 * num_total_attention_feat, 2 * num_total_attention_feat, kernel_size=1),
            nn.BatchNorm1d(2 * num_total_attention_feat),
            nn.ReLU(),
            nn.Conv1d(2 * num_total_attention_feat, num_total_attention_feat, kernel_size=1),
        )

        nn.init.constant_(self.post_merge[-1].bias, 0.0)

    def forward(self, init_query, key, value):
        batch_dim = init_query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) for l, x in
                             zip(self.proj, (init_query, key, value))]

        x, attention_weight = scaled_dot_product_attention(query, key, value)

        x = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        x = self.post_merge(torch.cat([x, init_query], dim=1))

        return x, attention_weight


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class LocalFeatureGather(nn.Module):
    def __init__(self, num_features, num_neighbor_markers):
        super(LocalFeatureGather, self).__init__()
        self.num_features = num_features
        self.num_neighbor_markers = num_neighbor_markers

        self.conv = nn.Sequential(nn.Conv2d(num_features * 2, num_features, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(num_features),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        graph_feature = get_graph_feature(x, self.num_neighbor_markers)
        x1 = self.conv(graph_feature)
        x2 = x1.max(dim=-1, keepdim=False)[0]

        return x2


class LayeredSelfAttention(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int, num_attention_heads: int = 4,
                 return_attention_weights=True, return_features=True, skip_connection=True):
        super(LayeredSelfAttention, self).__init__()
        self.self_attention_layers = nn.ModuleList([
            MultiHeadedAttention(num_attention_heads, feature_dim)
            for _ in range(num_layers)])
        self.return_attention_weights = return_attention_weights
        self.return_features = return_features
        self.skip_connection = skip_connection

    def forward(self, point_feats):
        """

        Parameters
        ----------
        point_feats: num_batch x num_points x num_feat

        Returns
        -------

        """
        all_attention_weights = []
        all_point_new_features = []

        for self_attention_layer in self.self_attention_layers:
            new_point_feats, attention_weight = self_attention_layer(point_feats, point_feats, point_feats)
            if self.skip_connection:
                point_feats = point_feats + new_point_feats
            else:
                point_feats = new_point_feats
            all_attention_weights.append(attention_weight)
            all_point_new_features.append(new_point_feats)

        return point_feats, torch.stack(all_attention_weights, axis=1), torch.stack(all_point_new_features, axis=1)


class LocalSelfAttention(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int, num_attention_heads: int, num_neighbors: int,
                 return_attention_weights=True, return_features=True, skip_connection=True):
        super(LocalSelfAttention, self).__init__()
        self.self_attention_layers = nn.ModuleList([
            MultiHeadedAttention(num_attention_heads, feature_dim)
            for _ in range(num_layers)])
        self.local_layers = nn.ModuleList([
            LocalFeatureGather(feature_dim, num_neighbors)
            for _ in range(num_layers)])
        self.num_layers = num_layers
        self.return_attention_weights = return_attention_weights
        self.return_features = return_features
        self.skip_connection = skip_connection

    def forward(self, point_feats):
        """

        Parameters
        ----------
        point_feats: num_batch x num_points x num_feat

        Returns
        -------

        """
        all_attention_weights = []
        all_point_new_features = []
        all_point_feat_local_dist = []
        for i in range(self.num_layers):
            self_attention_layer = self.self_attention_layers[i]
            local_feature_layer = self.local_layers[i]
            new_point_feats, attention_weight = self_attention_layer(point_feats, point_feats, point_feats)
            if self.skip_connection:
                point_feats = point_feats + new_point_feats
            else:
                point_feats = new_point_feats

            feat_local_dist = torch.cdist(point_feats.transpose(1, 2), point_feats.transpose(1, 2))
            new_point_feats = local_feature_layer(point_feats)
            if self.skip_connection:
                point_feats = point_feats + new_point_feats
            else:
                point_feats = new_point_feats

            all_point_feat_local_dist.append(feat_local_dist)
            all_attention_weights.append(attention_weight)
            all_point_new_features.append(new_point_feats)

        return (point_feats, torch.stack(all_attention_weights, axis=1),
                torch.stack(all_point_new_features, axis=1), torch.stack(all_point_feat_local_dist, axis=1))


def conv1d_layered(channels: list):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class ResDenseBlock(nn.Module):

    def __init__(self, num_feat_in, num_feat_out, num_h=256):
        super(ResDenseBlock, self).__init__()

        self.res_dense = nn.Sequential(
            nn.Linear(num_feat_in, num_h),
            nn.BatchNorm1d(num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_feat_out),
            nn.BatchNorm1d(num_feat_out)
        )

        self.res_dense_short = nn.Sequential(
            *([nn.Linear(num_feat_in, num_feat_out), nn.BatchNorm1d(num_feat_out)] if num_feat_in != num_feat_out else [
                nn.Identity()])
        )

    def forward(self, x):
        return self.res_dense(x) + self.res_dense_short(x)


class ResConv1DBlock(nn.Module):
    """ a series of 1D convolutions with residuals"""

    def __init__(self, num_feat_in, num_feat_out, num_h=256):
        super(ResConv1DBlock, self).__init__()

        self.res_conv1d = nn.Sequential(
            nn.Conv1d(num_feat_in, num_h, 1, 1),
            nn.BatchNorm1d(num_h),
            nn.ReLU(),
            nn.Conv1d(num_h, num_feat_out, 1, 1),
            nn.BatchNorm1d(num_feat_out)
        )

        self.res_conv1d_short = nn.Sequential(
            *([nn.Conv1d(num_feat_in, num_feat_out, 1, 1), nn.BatchNorm1d(num_feat_out)]
              if num_feat_in != num_feat_out else [nn.Identity()])
        )

    def forward(self, x):
        return self.res_conv1d(x) + self.res_conv1d_short(x)


class LayeredResConv1d(nn.Module):

    def __init__(self, num_feat_in, num_feat_out, num_layers, num_h=256, ):
        super(LayeredResConv1d, self).__init__()

        # self.res_conv1d_blocks = nn.ModuleList([
        #     nn.Sequential(ResConv1DBlock(num_feat_in, num_feat_out * 3, num_h=num_h),
        #                   nn.ReLU(),
        #                   nn.Conv1d(num_feat_out * 3, num_feat_out, kernel_size=1)
        #                   ) for _ in range(num_layers)])
        self.res_conv1d_blocks = nn.ModuleList([
            nn.Sequential(ResConv1DBlock(num_feat_in, num_feat_out * 3),
                          nn.ReLU(),
                          nn.Conv1d(num_feat_out * 3, num_feat_out, kernel_size=1),
                          nn.BatchNorm1d(num_feat_out),
                          nn.ReLU(),
                          nn.Conv1d(num_feat_out, num_feat_out, kernel_size=1),
                          ) for _ in range(num_layers)])

    def forward(self, point_feats):
        """

        Parameters
        ----------
        point_feats: num_batch x num_points x num_feat

        Returns
        -------

        """
        for res_conv1d_block in self.res_conv1d_blocks:
            new_point_feats = res_conv1d_block(point_feats)
            point_feats = point_feats + new_point_feats

        return point_feats


class Contiguous(nn.Module):
    def __init__(self):
        super(Contiguous, self).__init__()
        self._name = 'contiguous'

    def forward(self, x):
        return x.contiguous()


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.shape = args
        self._name = 'permute'

    def forward(self, x):
        return x.permute(self.shape)


class Transpose(nn.Module):
    def __init__(self, *args):
        super(Transpose, self).__init__()
        self.shape = args
        self._name = 'transpose'

    def forward(self, x):
        return x.transpose(*self.shape)


class SDivide(nn.Module):
    def __init__(self, scale):
        super(SDivide, self).__init__()
        self.scale = scale
        self._name = 'scalar_divide'

    def forward(self, x):
        return x / self.scale


class SelectItem(nn.Module):
    # https://stackoverflow.com/a/54660829
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
