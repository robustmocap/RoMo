import copy
import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn
from torch.nn import Embedding
from torch.nn import Transformer, TransformerEncoderLayer
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import *

import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


def MLP(dimensions: int, dropout: bool = False, batch_norm: bool = False, batch_norm_momentum: float = 1e-3):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Dropout(p=0.5) if dropout else nn.Identity(),
            nn.Linear(dimensions[i - 1], dimensions[i]),
            nn.ReLU(dimensions[i]),
            nn.BatchNorm1d(dimensions[i], affine=True, momentum=batch_norm_momentum) if batch_norm else nn.Identity())
        for i in range(1, len(dimensions))])


class MLPModel(nn.Module):
    def __init__(self, dim):
        super(MLPModel, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(MLP(self.dim[0:-1]),
                                    nn.Linear(self.dim[-2], self.dim[-1]))

    def forward(self, x):
        return self.layers(x)


def act_function(dim, func: str = "ReLU"):
    if func == 'ReLU':
        return nn.ReLU()
    elif func == "PReLU":
        return nn.PReLU()
    elif func == "LeakyReLU":
        return nn.LeakyReLU()
    elif func == "Hardswish":
        return nn.Hardswish()
    elif func == "Mish":
        return nn.Mish()
    return


class ResBlock(nn.Module):
    def __init__(self, dim: int, act: str = "ReLU", batch_norm: bool = False, dropout: bool = False):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.act = act
        self.layers = nn.Sequential(nn.Dropout(p=0.5) if dropout else nn.Identity(),
                                    nn.Linear(dim, dim),

                                    nn.BatchNorm1d(dim, affine=True, momentum=1e-3) if batch_norm else nn.Identity(),
                                    act_function(dim, func=self.act))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class SkeletonLinear(nn.Module):
    def __init__(self, in_channels, out_channels, node_num, edges):
        super(SkeletonLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edges = edges
        self.node_num = node_num

        self.weight = torch.zeros(out_channels * node_num, in_channels * node_num)
        self.mask = torch.zeros(out_channels * node_num, in_channels * node_num)
        self.bias = nn.Parameter(torch.Tensor(out_channels * node_num))

        for edge_idx in range(self.edges.shape[1]):
            edge = self.edges[:, edge_idx]
            tmp = torch.zeros((out_channels, in_channels))
            self.mask[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, x):
        weight_masked = self.weight * self.mask
        res = F.linear(x, weight_masked, self.bias)
        return res


class Marker2SkeletonLinear(nn.Module):
    def __init__(self, in_channels, out_channels, marker_num, joint_num, edges):
        super(Marker2SkeletonLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edges = edges
        self.joint_num = joint_num
        self.marker_num = marker_num

        self.weight = torch.zeros(out_channels * joint_num, in_channels * marker_num)
        self.mask = torch.zeros(out_channels * joint_num, in_channels * marker_num)
        self.bias = nn.Parameter(torch.Tensor(out_channels * joint_num))

        for edge_idx in range(self.edges.shape[1]):
            edge = self.edges[:, edge_idx]
            tmp = torch.zeros((out_channels, in_channels))
            self.mask[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[out_channels * edge[1]:out_channels * (edge[1] + 1),
            in_channels * edge[0]:in_channels * (edge[0] + 1)] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, x):
        weight_masked = self.weight * self.mask
        res = F.linear(x, weight_masked, self.bias)
        return res


class ResNetModel(nn.Module):
    def __init__(self, dim: list, act: str = "ReLU", batch_norm: bool = False, dropout: bool = False):
        super(ResNetModel, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.batch_norm = batch_norm
        self.act = act
        self.dropout = dropout
        for i in range(len(dim) - 1):
            if self.dim[i] == self.dim[i + 1]:
                self.layers.append(
                    ResBlock(self.dim[i], act=self.act[i], batch_norm=self.batch_norm, dropout=self.dropout))
                # self.layers.append(ResBlock(self.dim[i], batch_norm=self.batch_norm, dropout=self.dropout))
            elif i == len(dim) - 2:
                self.layers.append(nn.Sequential(nn.Linear(dim[-2], dim[-1])))
            else:
                self.layers.append(nn.Sequential(nn.Linear(dim[i], dim[i + 1]),
                                                 act_function(dim[i + 1], self.act[i]),
                                                 nn.BatchNorm1d(dim[i + 1], affine=True,
                                                                momentum=1e-3) if batch_norm else nn.Identity()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, feature_num = x.shape[0], x.shape[1], x.shape[2]
        x = x.view((batch_size * window_size, -1))
        for module in self.layers:
            x = module(x)
        x = x.view((batch_size, window_size, -1))
        return x


class HeteroGraphModel(nn.Module):
    def __init__(self, args, batch_norm: bool = False, dropout: bool = False):
        super(HeteroGraphModel, self).__init__()
        self.args = args
        self.use_window = args.DATA.use_window
        self.dim = args.NETWORK.network_dim
        self.marker_gcn_dim = args.NETWORK.marker_gcn_dim
        self.marker_gcn_act = args.NETWORK.marker_gcn_act
        self.skeleton_gcn_dim = args.NETWORK.skeleton_gcn_dim
        self.skeleton_gcn_act = args.NETWORK.skeleton_gcn_act
        self.marker_skeleton_conv_gcn_dim = args.NETWORK.marker_skeleton_conv_gcn_dim
        self.marker_skeleton_conv_gcn_act = args.NETWORK.marker_skeleton_conv_gcn_act

        self.batch_norm = batch_norm
        self.act = args.NETWORK.act_fun
        self.dropout = dropout
        self.marker_num = args.DATA.marker_num
        self.joint_num = args.DATA.joint_num
        self.marker_connect_mat = torch.from_numpy(
            np.load(os.path.join(args.DATA.root_dir, args.DATA.connect_mat_path)))
        self.marker_edges = dense_to_sparse(self.marker_connect_mat)[0]
        self.marker_edges = add_self_loops(self.marker_edges)[0]


        if self.args.DATA.rot_repr in ["mat_full", "mat_part", "quat"]:
            self.skin_weight = torch.from_numpy(
                np.load(os.path.join(args.DATA.root_dir, args.DATA.weight_mean_std_path))["mean"].T)
            if self.skin_weight.shape[0] == self.joint_num:
                self.skin_weight = self.skin_weight.T
            skin_weight_expand = torch.zeros((self.skin_weight.shape[0], self.skin_weight.shape[1] + 1))
            skin_weight_expand[:, :-1] = self.skin_weight
            skin_weight_expand[:, -1] = 1

            self.marker_skeleton_conv_gcn_list = nn.ModuleList()
            for i in range(len(self.marker_skeleton_conv_gcn_dim)):
                if i % 2 == 0:
                    self.marker_skeleton_conv_gcn_list.append(
                        Marker2SkeletonLinear(self.marker_skeleton_conv_gcn_dim[i][0],
                                              self.marker_skeleton_conv_gcn_dim[i][1],
                                              self.marker_num, self.joint_num + 1,
                                              np.vstack(
                                                  np.where(skin_weight_expand))))
                    self.marker_skeleton_conv_gcn_list.append(act_function(-1, self.marker_skeleton_conv_gcn_act[i]))
                else:
                    self.marker_skeleton_conv_gcn_list.append(
                        Marker2SkeletonLinear(self.marker_skeleton_conv_gcn_dim[i][0],
                                              self.marker_skeleton_conv_gcn_dim[i][1],
                                              self.joint_num + 1, self.marker_num,
                                              np.vstack(
                                                  np.where(skin_weight_expand.T))))
                    self.marker_skeleton_conv_gcn_list.append(act_function(-1, self.marker_skeleton_conv_gcn_act[i]))

            self.skeleton_parent_list = args.DATA.parent_list
            self.skeleton_connect_mat = np.zeros((len(self.skeleton_parent_list) + 1,
                                                  len(self.skeleton_parent_list) + 1))
            self.skeleton_connect_mat = torch.from_numpy(self.skeleton_connect_mat)

            for i in range(len(self.skeleton_parent_list)):
                self.skeleton_connect_mat[i, i] = 1
                self.skeleton_connect_mat[i, self.skeleton_parent_list[i]] = 1
                self.skeleton_connect_mat[self.skeleton_parent_list[i], i] = 1
            self.skeleton_connect_mat[-1] = 1
            self.skeleton_connect_mat[:, -1] = 1

            self.skeleton_edges = dense_to_sparse(self.skeleton_connect_mat)[0]

            self.marker_gcn_list = nn.ModuleList()
            for i in range(len(args.NETWORK.marker_gcn_dim) - 1):
                self.marker_gcn_list.append(
                    SkeletonLinear(args.NETWORK.marker_gcn_dim[i], args.NETWORK.marker_gcn_dim[i + 1],
                                   self.marker_num, self.marker_edges))
                self.marker_gcn_list.append(act_function(-1, self.marker_gcn_act[i]))

            self.skeleton_gcn_list = nn.ModuleList()
            for i in range(len(args.NETWORK.skeleton_gcn_dim) - 1):
                if i == 0:
                    self.skeleton_gcn_list.append(
                        SkeletonLinear(args.NETWORK.skeleton_gcn_dim[i] + self.marker_skeleton_conv_gcn_dim[-1][-1],
                                       args.NETWORK.skeleton_gcn_dim[i + 1], self.joint_num + 1,
                                       self.skeleton_edges))
                else:
                    self.skeleton_gcn_list.append(
                        SkeletonLinear(args.NETWORK.skeleton_gcn_dim[i], args.NETWORK.skeleton_gcn_dim[i + 1],
                                       self.joint_num + 1,
                                       self.skeleton_edges))
                if i == len(args.NETWORK.skeleton_gcn_dim) - 2:
                    pass
                else:
                    self.skeleton_gcn_list.append(act_function(-1, self.skeleton_gcn_act[i]))

            self.layers = nn.ModuleList()
            for i in range(len(self.dim) - 1):
                if self.dim[i] == self.dim[i + 1]:
                    self.layers.append(
                        ResBlock(self.dim[i], act=self.act[i], batch_norm=self.batch_norm, dropout=self.dropout))
                    # self.layers.append(ResBlock(self.dim[i], batch_norm=self.batch_norm, dropout=self.dropout))
                # elif i == len(self.dim) - 2:
                #     self.layers.append(nn.Sequential(nn.Linear(self.dim[-2], self.dim[-1])))
                else:
                    self.layers.append(nn.Sequential(nn.Linear(self.dim[i], self.dim[i + 1]),
                                                     act_function(self.dim[i + 1], self.act[i]),
                                                     nn.BatchNorm1d(self.dim[i + 1], affine=True,
                                                                    momentum=1e-3) if batch_norm else nn.Identity()))
        elif self.args.DATA.rot_repr == "joint_pos":
            self.skin_weight = torch.from_numpy(
                np.load(os.path.join(args.DATA.root_dir, args.DATA.weight_mean_std_path))["mean"].T)
            if self.skin_weight.shape[0] == self.joint_num:
                self.skin_weight = self.skin_weight.T
            skin_weight_expand = torch.zeros((self.skin_weight.shape[0], self.skin_weight.shape[1]))
            skin_weight_expand[:] = self.skin_weight

            self.marker_skeleton_conv_gcn_list = nn.ModuleList()
            for i in range(len(self.marker_skeleton_conv_gcn_dim)):
                if i % 2 == 0:
                    self.marker_skeleton_conv_gcn_list.append(
                        Marker2SkeletonLinear(self.marker_skeleton_conv_gcn_dim[i][0],
                                              self.marker_skeleton_conv_gcn_dim[i][1],
                                              self.marker_num, self.joint_num,
                                              np.vstack(np.where(skin_weight_expand))))
                    self.marker_skeleton_conv_gcn_list.append(act_function(-1, self.marker_skeleton_conv_gcn_act[i]))
                else:
                    self.marker_skeleton_conv_gcn_list.append(
                        Marker2SkeletonLinear(self.marker_skeleton_conv_gcn_dim[i][0],
                                              self.marker_skeleton_conv_gcn_dim[i][1],
                                              self.joint_num, self.marker_num,
                                              np.vstack(
                                                  np.where(skin_weight_expand.T))))
                    self.marker_skeleton_conv_gcn_list.append(act_function(-1, self.marker_skeleton_conv_gcn_act[i]))

            self.skeleton_parent_list = args.DATA.parent_list
            self.skeleton_connect_mat = np.zeros((len(self.skeleton_parent_list),
                                                  len(self.skeleton_parent_list)))
            self.skeleton_connect_mat = torch.from_numpy(self.skeleton_connect_mat)

            for i in range(len(self.skeleton_parent_list)):
                self.skeleton_connect_mat[i, i] = 1
                self.skeleton_connect_mat[i, self.skeleton_parent_list[i]] = 1
                self.skeleton_connect_mat[self.skeleton_parent_list[i], i] = 1

            self.skeleton_edges = dense_to_sparse(self.skeleton_connect_mat)[0]

            self.marker_gcn_list = nn.ModuleList()
            for i in range(len(args.NETWORK.marker_gcn_dim) - 1):
                self.marker_gcn_list.append(
                    SkeletonLinear(args.NETWORK.marker_gcn_dim[i], args.NETWORK.marker_gcn_dim[i + 1],
                                   self.marker_num, self.marker_edges))
                self.marker_gcn_list.append(act_function(-1, self.marker_gcn_act[i]))

            self.skeleton_gcn_list = nn.ModuleList()
            for i in range(len(args.NETWORK.skeleton_gcn_dim) - 1):
                if i == 0:
                    self.skeleton_gcn_list.append(
                        SkeletonLinear(args.NETWORK.skeleton_gcn_dim[i] + self.marker_skeleton_conv_gcn_dim[-1][-1],
                                       args.NETWORK.skeleton_gcn_dim[i + 1], self.joint_num,
                                       self.skeleton_edges))
                else:
                    self.skeleton_gcn_list.append(
                        SkeletonLinear(args.NETWORK.skeleton_gcn_dim[i], args.NETWORK.skeleton_gcn_dim[i + 1],
                                       self.joint_num,
                                       self.skeleton_edges))
                if i == len(args.NETWORK.skeleton_gcn_dim) - 2:
                    pass
                else:
                    self.skeleton_gcn_list.append(act_function(-1, self.skeleton_gcn_act[i]))

            self.layers = nn.ModuleList()
            for i in range(len(self.dim) - 1):
                if self.dim[i] == self.dim[i + 1]:
                    self.layers.append(
                        ResBlock(self.dim[i], act=self.act[i], batch_norm=self.batch_norm, dropout=self.dropout))
                    # self.layers.append(ResBlock(self.dim[i], batch_norm=self.batch_norm, dropout=self.dropout))
                # elif i == len(self.dim) - 2:
                #     self.layers.append(nn.Sequential(nn.Linear(self.dim[-2], self.dim[-1])))
                else:
                    self.layers.append(nn.Sequential(nn.Linear(self.dim[i], self.dim[i + 1]),
                                                     act_function(self.dim[i + 1], self.act[i]),
                                                     nn.BatchNorm1d(self.dim[i + 1], affine=True,
                                                                    momentum=1e-3) if batch_norm else nn.Identity()))
        else:
            raise NotImplementedError



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window_size, feature_num = x.shape[0], x.shape[1]
        x = x.view((window_size, -1))
        self.marker_edges = self.marker_edges.to(x.device)
        for i in range(len(self.marker_gcn_dim) - 1):
            x = self.marker_gcn_list[i * 2](x)
            x = self.marker_gcn_list[i * 2 + 1](x)

        joint_feature = None
        for i in range(len(self.marker_skeleton_conv_gcn_list)):
            if i == 0:
                joint_feature = self.marker_skeleton_conv_gcn_list[i](x)
            else:
                joint_feature = self.marker_skeleton_conv_gcn_list[i](joint_feature)

        for module in self.layers:
            x = module(x)

        x = torch.cat([x, joint_feature], dim=-1)
        for i in range(len(self.skeleton_gcn_list)):
            x = self.skeleton_gcn_list[i](x)

        if self.args.DATA.rot_repr in ["mat_full", "mat_part", "quat"]:
            x = x[:, :-(self.skeleton_gcn_dim[-1] - 3)]
        elif self.args.DATA.rot_repr == "joint_pos":
            pass
        else:
            raise NotImplementedError
        return x


class FKResNetModel(nn.Module):
    def __init__(self, args, encoder_dim: list, decoder_dim: list,
                 batch_norm: bool = False,
                 dropout: bool = False):
        super(FKResNetModel, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.encoder_layers = nn.ModuleList()
        self.motion_decoder = nn.ModuleList()
        self.joint_offset_decoder = nn.ModuleList()
        self.marker_config_decoder = nn.ModuleList()
        self.batch_norm = batch_norm
        self.encoder_act = args.encoder_act_fun
        self.decoder_act = args.decoder_act_fun
        self.dropout = dropout
        for i in range(len(self.encoder_dim) - 1):
            if self.encoder_dim[i] == self.encoder_dim[i + 1]:
                self.encoder_layers.append(
                    ResBlock(self.encoder_dim[i], act=self.encoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
            else:
                self.encoder_layers.append(nn.Sequential(nn.Linear(self.encoder_dim[i], self.encoder_dim[i + 1]),
                                                         act_function(self.encoder_dim[i + 1], self.encoder_act[i]),
                                                         nn.BatchNorm1d(self.encoder_dim[i + 1], affine=True,
                                                                        momentum=1e-3) if batch_norm else nn.Identity()))
        # self.encoder_layers.append(act_function(self.encoder_dim[-1], self.act))

        for i in range(len(self.decoder_dim) - 1):
            if self.decoder_dim[i] == self.decoder_dim[i + 1]:
                self.motion_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
                self.joint_offset_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
                self.marker_config_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
            else:
                self.motion_decoder.append(nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][0])))
                self.joint_offset_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][1])))
                self.marker_config_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][2])))

    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, window_size, feature_num = x.shape[0], x.shape[1], x.shape[2]
        latent = x.view((batch_size * window_size, -1))
        for module in self.encoder_layers:
            latent = module(latent)
        motion_latent = None
        marker_config_latent = None
        joint_offset_latent = None
        for module in self.motion_decoder:
            if motion_latent is None:
                motion_latent = module(latent)
            else:
                motion_latent = module(motion_latent)
        for module in self.joint_offset_decoder:
            if joint_offset_latent is None:
                joint_offset_latent = module(latent)
            else:
                joint_offset_latent = module(joint_offset_latent)
        for module in self.marker_config_decoder:
            if marker_config_latent is None:
                marker_config_latent = module(latent)
            else:
                marker_config_latent = module(marker_config_latent)

        motion_latent = motion_latent.view((batch_size, window_size, -1))
        joint_offset_latent = joint_offset_latent.view((batch_size, window_size, -1))
        marker_config_latent = marker_config_latent.view((batch_size, window_size, -1))

        return motion_latent, joint_offset_latent, marker_config_latent


class MoCapSolverFKResNetModel(nn.Module):
    def __init__(self, args, encoder_dim: list, decoder_dim: list,
                 batch_norm: bool = False,
                 dropout: bool = False):
        super(MoCapSolverFKResNetModel, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.encoder_layers = nn.ModuleList()
        self.motion_decoder = nn.ModuleList()
        self.joint_offset_decoder = nn.ModuleList()
        self.marker_config_decoder = nn.ModuleList()
        self.batch_norm = batch_norm
        self.encoder_act = args.encoder_act_fun
        self.decoder_act = args.decoder_act_fun
        self.dropout = dropout
        for i in range(len(self.encoder_dim) - 1):
            if self.encoder_dim[i] == self.encoder_dim[i + 1]:
                self.encoder_layers.append(
                    ResBlock(self.encoder_dim[i], act=self.encoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
            else:
                self.encoder_layers.append(
                    nn.Sequential(nn.Linear(self.encoder_dim[i], self.encoder_dim[i + 1]),
                                  act_function(self.encoder_dim[i + 1], self.encoder_act[i]),
                                  nn.BatchNorm1d(self.encoder_dim[i + 1], affine=True,
                                                 momentum=1e-3) if batch_norm else nn.Identity()))
        # self.encoder_layers.append(act_function(self.encoder_dim[-1], self.act))

        for i in range(len(self.decoder_dim) - 1):
            if self.decoder_dim[i] == self.decoder_dim[i + 1]:
                self.motion_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
                self.joint_offset_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
                self.marker_config_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
            else:
                self.motion_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][0])))
                self.joint_offset_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][1])))
                self.marker_config_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][2])))

    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, window_size, feature_num = x.shape[0], x.shape[1], x.shape[2]
        latent = x.view((batch_size * window_size, -1))

        for module in self.encoder_layers:
            latent = module(latent)
        motion_latent = None
        marker_config_latent = None
        joint_offset_latent = None
        for module in self.motion_decoder:
            if motion_latent is None:
                motion_latent = module(latent)
            else:
                motion_latent = module(motion_latent)
        for module in self.joint_offset_decoder:
            if joint_offset_latent is None:
                joint_offset_latent = module(latent)
            else:
                joint_offset_latent = module(joint_offset_latent)
        for module in self.marker_config_decoder:
            if marker_config_latent is None:
                marker_config_latent = module(latent)
            else:
                marker_config_latent = module(marker_config_latent)

        motion_latent = motion_latent.view((batch_size, window_size, -1))
        joint_offset_latent = joint_offset_latent.view((batch_size, -1))
        marker_config_latent = marker_config_latent.view((batch_size, -1))

        return motion_latent, joint_offset_latent, marker_config_latent


class FKResNetModelwithJointPos(nn.Module):
    def __init__(self, args, encoder_dim: list, decoder_dim: list,
                 batch_norm: bool = False, dropout: bool = False):
        super(FKResNetModelwithJointPos, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.encoder_layers = nn.ModuleList()
        self.motion_decoder = nn.ModuleList()
        self.joint_offset_decoder = nn.ModuleList()
        self.marker_config_decoder = nn.ModuleList()
        self.joint_pos_decoder = nn.ModuleList()
        self.batch_norm = batch_norm
        self.encoder_act = args.encoder_act_fun
        self.decoder_act = args.decoder_act_fun
        self.dropout = dropout
        for i in range(len(self.encoder_dim) - 1):
            if self.encoder_dim[i] == self.encoder_dim[i + 1]:
                self.encoder_layers.append(
                    ResBlock(self.encoder_dim[i], act=self.encoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
            else:
                self.encoder_layers.append(nn.Sequential(nn.Linear(self.encoder_dim[i], self.encoder_dim[i + 1]),
                                                         act_function(self.encoder_dim[i + 1], self.encoder_act[i]),
                                                         nn.BatchNorm1d(self.encoder_dim[i + 1], affine=True,
                                                                        momentum=1e-3) if batch_norm else nn.Identity()))
        # self.encoder_layers.append(act_function(self.encoder_dim[-1], self.act))

        for i in range(len(self.decoder_dim) - 1):
            if self.decoder_dim[i] == self.decoder_dim[i + 1]:
                self.motion_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
                self.joint_offset_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
                self.marker_config_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
                self.joint_pos_decoder.append(
                    ResBlock(self.decoder_dim[i], act=self.decoder_act[i], batch_norm=self.batch_norm,
                             dropout=self.dropout))
            else:
                self.motion_decoder.append(nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][0])))
                self.joint_offset_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][1])))
                self.marker_config_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][2])))
                self.joint_pos_decoder.append(
                    nn.Sequential(nn.Linear(self.decoder_dim[-2], self.decoder_dim[-1][3])))

    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, window_size, feature_num = x.shape[0], x.shape[1], x.shape[2]
        latent = x.view((batch_size * window_size, -1))
        for module in self.encoder_layers:
            latent = module(latent)
        motion_latent = None
        marker_config_latent = None
        joint_offset_latent = None
        joint_pos_latent = None
        for module in self.motion_decoder:
            if motion_latent is None:
                motion_latent = module(latent)
            else:
                motion_latent = module(motion_latent)
        for module in self.joint_offset_decoder:
            if joint_offset_latent is None:
                joint_offset_latent = module(latent)
            else:
                joint_offset_latent = module(joint_offset_latent)
        for module in self.marker_config_decoder:
            if marker_config_latent is None:
                marker_config_latent = module(latent)
            else:
                marker_config_latent = module(marker_config_latent)
        for module in self.joint_pos_decoder:
            if joint_pos_latent is None:
                joint_pos_latent = module(latent)
            else:
                joint_pos_latent = module(joint_pos_latent)

        motion_latent = motion_latent.view((batch_size, window_size, -1))
        joint_offset_latent = joint_offset_latent.view((batch_size, window_size, -1))
        marker_config_latent = marker_config_latent.view((batch_size, window_size, -1))
        joint_pos_latent = joint_pos_latent.view((batch_size, window_size, -1))

        return motion_latent, joint_offset_latent, marker_config_latent, joint_pos_latent


class MotionJointPosModel(nn.Module):
    def __init__(self, args, network_dim: list, batch_norm: bool = False, dropout: bool = False):
        super(MotionJointPosModel, self).__init__()
        self.network_dim = network_dim
        self.act = args.MODEL.act_fun
        self.motion_decoder = nn.ModuleList()
        self.joint_pos_decoder = nn.ModuleList()
        self.batch_norm = batch_norm
        self.dropout = dropout
        for i in range(len(self.network_dim) - 1):
            if self.network_dim[i] == self.network_dim[i + 1]:
                self.motion_decoder.append(
                    ResBlock(self.network_dim[i], act=self.act[i], batch_norm=self.batch_norm, dropout=self.dropout))
                # self.layers.append(ResBlock(self.dim[i], batch_norm=self.batch_norm, dropout=self.dropout))
            elif i == len(self.network_dim) - 2:
                self.motion_decoder.append(nn.Sequential(nn.Linear(self.network_dim[-2], self.network_dim[-1][0])))
            else:
                self.motion_decoder.append(nn.Sequential(nn.Linear(self.network_dim[i], self.network_dim[i + 1]),
                                                         act_function(self.network_dim[i + 1], self.act[i]),
                                                         nn.BatchNorm1d(self.network_dim[i + 1], affine=True,
                                                                        momentum=1e-3) if batch_norm else nn.Identity()))
        for i in range(len(self.network_dim) - 1):
            if self.network_dim[i] == self.network_dim[i + 1]:
                self.joint_pos_decoder.append(
                    ResBlock(self.network_dim[i], act=self.act[i], batch_norm=self.batch_norm, dropout=self.dropout))
                # self.layers.append(ResBlock(self.dim[i], batch_norm=self.batch_norm, dropout=self.dropout))
            elif i == len(self.network_dim) - 2:
                self.joint_pos_decoder.append(nn.Sequential(nn.Linear(self.network_dim[-2], self.network_dim[-1][1])))
            else:
                self.joint_pos_decoder.append(nn.Sequential(nn.Linear(self.network_dim[i], self.network_dim[i + 1]),
                                                            act_function(self.network_dim[i + 1], self.act[i]),
                                                            nn.BatchNorm1d(self.network_dim[i + 1], affine=True,
                                                                           momentum=1e-3) if batch_norm else nn.Identity()))

    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, window_size, feature_num = x.shape[0], x.shape[1], x.shape[2]
        x = x.view((batch_size * window_size, -1))
        motion_latent = None
        joint_pos_latent = None
        for module in self.motion_decoder:
            if motion_latent == None:
                motion_latent = module(x)
            else:
                motion_latent = module(motion_latent)

        for module in self.joint_pos_decoder:
            if joint_pos_latent == None:
                joint_pos_latent = module(x)
            else:
                joint_pos_latent = module(joint_pos_latent)

        motion_latent = motion_latent.view((batch_size, window_size, -1))
        joint_pos_latent = joint_pos_latent.view((batch_size, window_size, -1))

        return motion_latent, joint_pos_latent


class EncoderDecoder(nn.Module):
    def __init__(self, dim, dropout, batch_norm=True, batch_norm_momentum=1e-3, use_relu6=True):
        super(EncoderDecoder, self).__init__()
        self.dim = dim
        self.use_relu6 = use_relu6
        # dropout is forgotten?
        self.dropout = dropout
        self.dropout = nn.Dropout(p=dropout)

        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        # print(self.dim)

        self.encoder_layers = nn.Sequential(self.encoder_MLP(self.dim[:-1], batch_norm=self.batch_norm),
                                            nn.Linear(self.dim[-2], self.dim[-1]),
                                            nn.BatchNorm1d(self.dim[-1], affine=True,
                                                           momentum=self.batch_norm_momentum) if self.batch_norm
                                            else nn.Identity(),
                                            nn.ReLU6() if self.use_relu6 else nn.PReLU())
        self.decoder_layers = nn.Sequential(self.decoder_MLP(self.dim[:0:-1], batch_norm=self.batch_norm),
                                            nn.Linear(self.dim[1], self.dim[0]),
                                            nn.PReLU(self.dim[0]))

    def encoder_MLP(self, dimensions, batch_norm=False, batch_norm_momentum=1e-3):
        return nn.Sequential(*[
            nn.Sequential(
                nn.BatchNorm1d(dimensions[i - 1], affine=True,
                               momentum=batch_norm_momentum) if batch_norm else nn.Identity(),
                nn.Linear(dimensions[i - 1], dimensions[i]),
                nn.PReLU(dimensions[i]))
            for i in range(1, len(dimensions))])

    def decoder_MLP(self, dimensions, batch_norm=False, batch_norm_momentum=1e-3):
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dimensions[i - 1], dimensions[i]),
                nn.PReLU(dimensions[i]),
                nn.BatchNorm1d(dimensions[i], affine=True,
                               momentum=batch_norm_momentum) if batch_norm else nn.Identity())
            for i in range(1, len(dimensions))])

    def encode(self, x):
        return self.encoder_layers(x)

    def decode(self, x):
        return self.decoder_layers(x)


class ResNetBiLSTM(nn.Module):
    def __init__(self, dim, batch_norm=False, layer_num=1):
        super(ResNetBiLSTM, self).__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.res_net_layers = nn.ModuleList()
        for i in range(len(dim) - 3):
            if self.dim[i] == self.dim[i + 1]:
                self.res_net_layers.append(ResBlock(self.dim[i]))
            else:
                self.res_net_layers.append(
                    nn.Sequential(nn.Linear(dim[i], dim[i + 1]),
                                  nn.ReLU(dim[i + 1]),
                                  nn.BatchNorm1d(dim[i + 1], affine=True, momentum=1e-3)
                                  if batch_norm else nn.Identity()))

        self.lstm_input_num = self.dim[-3]
        self.lstm_hidden_num = self.dim[-2]
        self.output_num = self.dim[-1]

        self.lstm = nn.LSTM(input_size=self.lstm_input_num, hidden_size=self.lstm_hidden_num,
                            num_layers=self.layer_num,
                            bidirectional=True, batch_first=True)
        # since it is a BI-LSTM, the hidden dim has to be multiplied 2
        self.out_linear = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(self.lstm_hidden_num * 2, self.output_num),
                                        nn.PReLU(self.output_num),
                                        nn.BatchNorm1d(self.output_num),
                                        nn.Linear(self.output_num, self.output_num))

    def forward(self, x, hidden=None, c=None):
        if hidden is None:
            device = self.lstm.all_weights[0][0].device
            hidden = torch.zeros(self.layer_num * 2, x.shape[0], self.lstm_hidden_num).to(device)
            c = torch.zeros(self.layer_num * 2, x.shape[0], self.lstm_hidden_num).to(device)
        batch_size, window_size = x.shape[0], x.shape[1]
        # x = torch.transpose(x, 0, 1)
        for module in self.res_net_layers:
            x = module(x)
        y, (hidden, c) = self.lstm(x, (hidden, c))

        out = self.out_linear(y.reshape((batch_size * window_size, -1)))
        out = out.reshape((batch_size, window_size, -1))
        # out = torch.transpose(out, 0, 1)
        return out, (hidden, c)


class ResNetTCN(nn.Module):
    def __init__(self, dim, tcn_kernel_size, batch_norm=False):
        super(ResNetTCN, self).__init__()
        self.start_dim = dim[0]
        self.dim = dim[1:-1]
        self.last_dim = dim[-1]
        self.tcn_kernel_size = tcn_kernel_size
        self.res_net_layers = nn.ModuleList()
        for i in range(len(self.dim) - 1):
            if self.dim[i] == self.dim[i + 1]:
                self.res_net_layers.append(ResBlock(self.dim[i]))
            else:
                self.res_net_layers.append(nn.Sequential(nn.Linear(self.dim[i], self.dim[i + 1]),
                                                         nn.ReLU(self.dim[i + 1]),
                                                         nn.BatchNorm1d(self.dim[i + 1], affine=True,
                                                                        momentum=1e-3) if batch_norm else nn.Identity()))
        # self.last_layers = nn.Linear(self.dim[-1], self.last_dim)
        self.start_tcn_layer = nn.Conv1d(self.start_dim, self.dim[0], self.tcn_kernel_size,
                                         padding_mode="replicate", padding=self.tcn_kernel_size // 2)
        self.start_act_layer = nn.ReLU(self.dim[0])
        self.last_linear_layer = nn.Linear(self.dim[-1], self.last_dim)
        # print()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.start_tcn_layer(x)
        x = self.start_act_layer(x)

        x = x.permute(0, 2, 1)

        for module in self.res_net_layers:
            x = module(x)

        x = self.last_linear_layer(x)
        # x = self.last_layers(x)
        # print()
        return x


class BiLSTM(nn.Module):
    def __init__(self, args, input_num: int, hidden_num: int, output_num: int,
                 act: str = "ReLU", layer_num: int = 1):
        super(BiLSTM, self).__init__()
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.layer_num = layer_num
        self.act = act
        self.lstm = nn.LSTM(input_size=input_num, hidden_size=hidden_num, num_layers=layer_num,
                            bidirectional=True, batch_first=True)
        # since it is a BI-LSTM, the hidden dim has to be multiplied 2
        self.out_linear = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_num * 2, output_num),
                                        act_function(output_num, self.act),
                                        nn.BatchNorm1d(output_num),
                                        nn.Linear(output_num, output_num))

    def forward(self, x, hidden=None, c=None):
        if hidden is None:
            device = self.lstm.all_weights[0][0].device
            hidden = torch.zeros(self.layer_num * 2, x.shape[0], self.hidden_num).to(device)
            c = torch.zeros(self.layer_num * 2, x.shape[0], self.hidden_num).to(device)
        batch_size, window_size = x.shape[0], x.shape[1]
        # x = torch.transpose(x, 0, 1)
        x = x.view((batch_size, window_size, -1))
        y, (hidden, c) = self.lstm(x, (hidden, c))

        out = self.out_linear(y.reshape((batch_size * window_size, -1)))
        out = out.reshape((batch_size, window_size, -1))
        # out = torch.transpose(out, 0, 1)
        return out, (hidden, c)


class GNNBiLSTM(nn.Module):
    def __init__(self, args, input_num: int, hidden_num: int, output_num: int,
                 act: str = "ReLU", layer_num: int = 1):
        super(GNNBiLSTM, self).__init__()
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.layer_num = layer_num
        self.act = act
        self.marker_num = args.DATA.marker_num
        self.marker_gcn_act = args.NETWORK.marker_gcn_act

        self.marker_connect_mat = torch.from_numpy(
            np.load(os.path.join(args.DATA.root_dir, args.DATA.connect_mat_path)))
        self.marker_edges = dense_to_sparse(self.marker_connect_mat)[0]
        self.marker_edges = add_self_loops(self.marker_edges)[0]

        self.marker_gcn_list = nn.ModuleList()
        for i in range(len(args.NETWORK.marker_gcn_dim) - 1):
            self.marker_gcn_list.append(SkeletonLinear(args.NETWORK.marker_gcn_dim[i], args.MODEL.marker_gcn_dim[i + 1],
                                                       self.marker_num, self.marker_edges))
            self.marker_gcn_list.append(act_function(-1, self.marker_gcn_act[i]))

        self.lstm = nn.LSTM(input_size=args.NETWORK.marker_gcn_dim[-1] * self.marker_num,
                            hidden_size=hidden_num,
                            num_layers=layer_num,
                            bidirectional=True, batch_first=True)
        # since it is a BI-LSTM, the hidden dim has to be multiplied 2
        self.out_linear = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_num * 2, output_num),
                                        act_function(output_num, self.act),
                                        # nn.BatchNorm1d(output_num),
                                        nn.Linear(output_num, output_num))

    def forward(self, x, hidden=None, c=None):
        if hidden is None:
            device = self.lstm.all_weights[0][0].device
            hidden = torch.zeros(self.layer_num * 2, x.shape[0], self.hidden_num).to(device)
            c = torch.zeros(self.layer_num * 2, x.shape[0], self.hidden_num).to(device)
        batch_size, window_size = x.shape[0], x.shape[1]
        # x = torch.transpose(x, 0, 1)
        for layer in self.marker_gcn_list:
            x = layer(x.view(batch_size * window_size, -1))

        x = x.view((batch_size, window_size, -1))
        y, (hidden, c) = self.lstm(x, (hidden, c))

        out = self.out_linear(y.reshape((batch_size * window_size, -1)))
        out = out.reshape((batch_size, window_size, -1))
        # out = torch.transpose(out, 0, 1)
        return out, (hidden, c)


class AttentionLSTM(nn.Module):
    def __init__(self, input_num, hidden_num, window_size):
        super(AttentionLSTM, self).__init__()
        self.window_size = window_size
        self.input_num = input_num
        self.hidden_num = hidden_num

        self.cell = nn.LSTMCell(input_size=input_num, hidden_size=hidden_num)
        self.W = nn.Linear(self.hidden_num * 2, self.window_size)
        self.U = nn.Linear(self.input_num, self.window_size)
        self.v = nn.Linear(self.window_size, 1)

    def forward(self, x, hidden=None, c=None):
        batch_size, window_size = x.shape[0], x.shape[1]
        if hidden is None:
            device = self.cell.bias_hh.device
            hidden = torch.zeros((batch_size, self.hidden_num)).to(device)
            c = torch.zeros((batch_size, self.hidden_num)).to(device)

        hidden_list = []
        c_list = []
        for t in range(window_size):
            e = torch.unsqueeze(self.W(torch.hstack([hidden, c])), 1) + \
                self.U(x.view((batch_size * window_size, -1))).view((batch_size, window_size, -1))
            e = self.v(torch.tanh(e).view((batch_size * window_size, -1))).view((batch_size, window_size))
            alpha = torch.exp(e) / torch.sum(torch.exp(e), dim=1, keepdim=True)
            weighted_x = torch.einsum("ab,abc->ac", alpha, x)
            hidden, c = self.cell(weighted_x, (hidden, c))
            hidden_list.append(hidden)
            c_list.append(c)
        hidden_list = torch.cat([torch.unsqueeze(h, 1) for h in hidden_list], 1)
        c_list = torch.cat([torch.unsqueeze(c, 1) for c in c_list], 1)
        return hidden_list, c_list


class BiAttentionLSTM(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, window_size, layer_num=1):
        super(BiAttentionLSTM, self).__init__()
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.layer_num = layer_num
        self.output_num = output_num
        self.window_size = window_size
        self.encoder_forward_attention_cell = AttentionLSTM(self.input_num, self.hidden_num, self.window_size)
        self.encoder_backward_attention_cell = AttentionLSTM(self.input_num, self.hidden_num, self.window_size)

        self.decoder_forward_attention_cell = AttentionLSTM(self.hidden_num * 2, self.hidden_num, self.window_size)
        self.decoder_backward_attention_cell = AttentionLSTM(self.hidden_num * 2, self.hidden_num, self.window_size)

        self.out_layer = nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(hidden_num * 2, output_num),
                                       nn.PReLU(output_num),
                                       nn.BatchNorm1d(output_num),
                                       nn.Linear(output_num, output_num))

    def forward(self, x, hidden=None, c=None):
        batch_size, window_size = x.shape[0], x.shape[1]

        encoder_forward_hidden_list, _ = self.encoder_forward_attention_cell(x)
        encoder_backward_hidden_list, _ = self.encoder_backward_attention_cell(torch.flip(x, dims=[1]))

        hidden = torch.cat([encoder_forward_hidden_list, torch.flip(encoder_backward_hidden_list, [1])], 2)

        decoder_forward_out_list, _ = self.decoder_forward_attention_cell(hidden)
        decoder_backward_out_list, _ = self.decoder_backward_attention_cell(torch.flip(hidden, dims=[1]))

        out = torch.cat([decoder_forward_out_list, torch.flip(decoder_backward_out_list, [1])], 2)

        out = self.out_layer(out.view((batch_size * window_size, -1))).view((batch_size, window_size, -1))
        return out, 1


class TransformerModel(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, window_size):
        super(TransformerModel, self).__init__()
        self.in_linear = nn.Sequential(nn.Linear(input_num, hidden_num),
                                       nn.PReLU(hidden_num),
                                       nn.BatchNorm1d(hidden_num),
                                       nn.Linear(hidden_num, hidden_num))
        self.out_linear = nn.Sequential(nn.Linear(hidden_num, output_num),
                                        nn.PReLU(output_num),
                                        nn.BatchNorm1d(output_num),
                                        nn.Linear(output_num, output_num))
        self.transformer = TransformerEncoderLayer(hidden_num, 8, batch_first=True)
        self.window_size = window_size

    def forward(self, x):
        batch_size, window_size = x.shape[0], x.shape[1]

        x = self.in_linear(x.clone().reshape((batch_size * window_size, -1))).reshape((batch_size, window_size, -1))
        x = self.transformer(x.clone())
        out = self.out_linear(x.clone().reshape((batch_size * window_size, -1))).reshape((batch_size, window_size, -1))
        return out
