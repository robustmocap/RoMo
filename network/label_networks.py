import copy
import os
import math
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn
from torch.nn import Embedding
from torch.nn import Transformer, TransformerEncoderLayer

from torch_geometric.utils import *
from .model_components import *
from .optimal_transport import *


class SOMAScorePredictor(nn.Module):
    def __init__(self, args):
        super(SOMAScorePredictor, self).__init__()
        self.args = args

        attention_feat_dim = args.NETWORK.attention_feat_dim
        attention_layer_num = args.NETWORK.attention_layer_num
        attention_head_num = args.NETWORK.attention_head_num
        label_num = args.DATA.label_num

        self.pre_layers = nn.Sequential(
            Transpose(-2, -1),
            ResConv1DBlock(3, attention_feat_dim),
            nn.ReLU(), )
        self.point_attention = LayeredSelfAttention(attention_feat_dim,
                                                    attention_layer_num,
                                                    attention_head_num)
        self.post_layers = nn.Sequential(
            nn.ReLU(),
            ResConv1DBlock(attention_feat_dim, attention_feat_dim * 2),
            nn.ReLU(),
            SDivide(attention_feat_dim ** .5),
            nn.Conv1d(attention_feat_dim * 2, label_num, 1, 1),
            Transpose(-2, -1), )

        if self.args.DATA.train_part == "full_body":
            self.part_layers = nn.Sequential(
                nn.ReLU(),
                ResConv1DBlock(attention_feat_dim, attention_feat_dim * 2),
                nn.ReLU(),
                SDivide(attention_feat_dim ** .5),
                nn.Conv1d(attention_feat_dim * 2, 3, 1, 1),
                Transpose(-2, -1), )

    def forward(self, pts_centered):
        out_dict = {}

        x = self.pre_layers(pts_centered)

        att_res, att_weights, new_features = self.point_attention(x)
        out_dict["scores"] = self.post_layers(att_res)
        out_dict["attention_weights"] = att_weights
        out_dict["point_features"] = new_features.transpose(-1, -2)

        if self.args.DATA.train_part == "full_body":
            out_dict["part_scores"] = self.part_layers(att_res)

        return out_dict


class SOMA_Network(nn.Module):
    def __init__(self, args):
        super(SOMA_Network, self).__init__()
        self.args = args

        self.score_predictor = SOMAScorePredictor(args)

        self.bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter('bin_score', self.bin_score)
        self.num_sinkhorn_iters = args.NETWORK.num_sinkhorn_iters

    def forward(self, pts):
        score_predictor_res = self.score_predictor(pts)

        attention_weights = score_predictor_res["attention_weights"]
        scores = score_predictor_res["scores"]
        aug_asmat = log_optimal_transport(scores, self.bin_score, iters=self.num_sinkhorn_iters)

        output_dict = {"attention_weights": attention_weights,
                       "scores": scores,
                       'label_ids': aug_asmat[:, :-1].argmax(-1),
                       'label_confidence': aug_asmat[:, :-1].exp(),
                       'aug_asmat': aug_asmat, }
        if self.args.DATA.train_part == "full_body":
            part_scores = score_predictor_res["part_scores"]
            output_dict["part_scores"] = part_scores
            output_dict["part_ids"] = part_scores.argmax(-1)
        return output_dict


class ScorePredictor(nn.Module):
    def __init__(self, args):
        super(ScorePredictor, self).__init__()
        self.args = copy.deepcopy(args)

        attention_feat_dim = args.NETWORK.attention_feat_dim
        attention_layer_num = args.NETWORK.attention_layer_num
        attention_head_num = args.NETWORK.attention_head_num
        neighbor_marker_num = args.NETWORK.neighbor_marker_num
        skip_connection = args.NETWORK.skip_connection
        label_num = args.DATA.label_num

        self.pre_layers = nn.Sequential(
            Transpose(-2, -1),
            ResConv1DBlock(3, attention_feat_dim),
            nn.ReLU(), )

        self.mid_layers = LocalSelfAttention(attention_feat_dim,
                                             attention_layer_num,
                                             attention_head_num,
                                             neighbor_marker_num,
                                             skip_connection=skip_connection)
        self.post_layers = nn.Sequential(
            nn.ReLU(),
            ResConv1DBlock(attention_feat_dim, attention_feat_dim * 2),
            nn.ReLU(),
            SDivide(attention_feat_dim ** .5),
            nn.Conv1d(attention_feat_dim * 2, label_num, 1, 1),
            Transpose(-2, -1), )

        if self.args.DATA.train_part == "full_body":
            self.part_layers = nn.Sequential(
                nn.ReLU(),
                ResConv1DBlock(attention_feat_dim, attention_feat_dim * 2),
                nn.ReLU(),
                SDivide(attention_feat_dim ** .5),
                nn.Conv1d(attention_feat_dim * 2, 3, 1, 1),
                Transpose(-2, -1), )

    def forward(self, pts_centered):
        out_dict = {}

        x = self.pre_layers(pts_centered)

        att_res, att_weights, new_features, feature_local_dist = self.mid_layers(x)
        out_dict["scores"] = self.post_layers(att_res)
        out_dict["attention_weights"] = att_weights[::16].detach().clone()
        out_dict["point_features"] = new_features.transpose(-1, -2)
        out_dict["feature_local_dist"] = feature_local_dist[::16].detach().clone()

        if self.args.DATA.train_part == "full_body":
            out_dict["part_scores"] = self.part_layers(att_res)

        return out_dict


class PCT_Labeller(nn.Module):
    def __init__(self, args):
        super(PCT_Labeller, self).__init__()
        self.args = args

        self.score_predictor = ScorePredictor(args)

        self.bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter('bin_score', self.bin_score)
        self.num_sinkhorn_iters = args.NETWORK.num_sinkhorn_iters

    def forward(self, pts):
        score_predictor_res = self.score_predictor(pts)

        attention_weights = score_predictor_res["attention_weights"]
        point_features = score_predictor_res["point_features"]
        scores = score_predictor_res["scores"]
        feature_local_dist = score_predictor_res["feature_local_dist"]
        aug_asmat = log_optimal_transport(scores, self.bin_score, iters=self.num_sinkhorn_iters)

        output_dict = {"feature_local_dist": feature_local_dist,
                       "attention_weights": attention_weights,
                       "point_features": point_features,
                       "scores": scores,
                       'label_ids': aug_asmat[:, :-1].argmax(-1),
                       'label_confidence': aug_asmat[:, :-1].exp(),
                       'aug_asmat': aug_asmat, }
        if self.args.DATA.train_part == "full_body":
            part_scores = score_predictor_res["part_scores"]
            output_dict["part_scores"] = part_scores
            output_dict["part_ids"] = part_scores.argmax(-1)
        return output_dict