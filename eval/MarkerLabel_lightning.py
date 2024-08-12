import copy
import time

from network.networks import *
from utils.utils import *
from utils.tracklet import *

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import network
import dataloader
import lightning_module

from utils.parser_util import parse_args_from_yaml
from utils.utils import (add_marker_global_transform, remove_marker_global_transform,
                         compute_offsets_np)


def main():
    # torch.autograd.set_detect_anomaly(True)

    full_body_marker_path = "samples/label/full_body.npy"
    body_marker_path = "samples/label/body.npy"
    left_hand_marker_path = "samples/label/left_hand.npy"
    right_hand_marker_path = "samples/label/right_hand.npy"

    body_marker_res_path = "samples/label/labeled_body.npy"
    left_hand_marker_res_path = "samples/label/labeled_left_hand.npy"
    right_hand_marker_res_path = "samples/label/labeled_right_hand.npy"

    body_marker_num = 53
    hand_marker_num = 16

    args = parse_args_from_yaml()
    pl.seed_everything(args.SEED)

    if args.DATA.train_part == "full_body":
        input_file_path = full_body_marker_path
    elif args.DATA.train_part == "body":
        input_file_path = body_marker_path
        out_file_path = body_marker_res_path
    elif args.DATA.train_part == "left_hand":
        input_file_path = left_hand_marker_path
        out_file_path = left_hand_marker_res_path
    elif args.DATA.train_part == "right_hand":
        input_file_path = right_hand_marker_path
        out_file_path = right_hand_marker_res_path

    net = network.build_network(args)

    module = lightning_module.build_lightning_module(args, net, None)

    net = net.to("cuda:0")
    net.eval()

    marker_pos = np.load(input_file_path)
    input_pts = marker_pos

    time_length, marker_num = input_pts.shape[0], input_pts.shape[1]
    shuffled_pts = copy.deepcopy(input_pts)
    for i in range(time_length):
        perm_vec = np.random.permutation(marker_num)
        shuffled_pts[i] = input_pts[i][perm_vec]
    input_pts = torch.from_numpy(copy.deepcopy(shuffled_pts))

    input_pts = input_pts.to("cuda:0")
    input_pts = input_pts * args.DATA.scale

    offset = compute_offsets_np(input_pts.cpu().numpy())
    input_pts = input_pts - torch.from_numpy(offset).to(input_pts.device)

    out = net(input_pts)

    if args.DATA.train_part == "full_body":
        # segmentation
        part_score = out["part_scores"]

        body_marker_ids = part_score[:, :, 0].topk(body_marker_num, dim=1)[1]
        left_hand_marker_ids = part_score[:, :, 1].topk(hand_marker_num, dim=1)[1]
        right_hand_marker_ids = part_score[:, :, 2].topk(hand_marker_num, dim=1)[1]

        body_markers = torch.stack([input_pts[i, body_marker_ids[i]] for i in range(time_length)])
        left_hand_markers = torch.stack([input_pts[i, left_hand_marker_ids[i]] for i in range(time_length)])
        right_hand_markers = torch.stack([input_pts[i, right_hand_marker_ids[i]] for i in range(time_length)])

        np.save("seg_" + body_marker_path, body_markers.cpu().numpy())
        np.save("seg_" + left_hand_marker_path, left_hand_markers.cpu().numpy())
        np.save("seg_" + right_hand_marker_path, right_hand_markers.cpu().numpy())
    else:
        # classification
        if args.EVAL.use_tracklet:
            input_pts = input_pts.cpu().numpy()
            label_confidence = out["label_confidence"].detach().cpu().numpy()
            point_features = out["point_features"].detach().cpu().numpy()
            tracklet_label_ids = tracklet_labeling(args.EVAL.tracklet, label_confidence, input_pts, point_features)
            tracklet_label_ids = torch.from_numpy(tracklet_label_ids)

            res = np.zeros((time_length, marker_num, 3))
            for frame in range(time_length):
                for i, marker_idx in enumerate(tracklet_label_ids[frame]):
                    if marker_idx < marker_num:
                        res[frame, marker_idx] = shuffled_pts[frame, i]
            np.save(out_file_path, res)

            # print()
        else:
            res_ids = out["label_ids"]
            res = np.zeros((time_length, marker_num, 3))
            for frame in range(time_length):
                for i, marker_idx in enumerate(res_ids[frame]):
                    if marker_idx < marker_num:
                        res[frame, marker_idx] = shuffled_pts[frame, i]
            np.save(out_file_path, res)


if __name__ == "__main__":
    main()
