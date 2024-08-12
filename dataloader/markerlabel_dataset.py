import copy
import json
import os

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.utils import (add_marker_global_transform, remove_marker_global_transform,
                         compute_offsets_np)


class MarkerLabelDataset(Dataset):
    def __init__(self, args, h5_file_path: str, file_list_path: str, marker_num: int,
                 window_size: int, train_part: str,
                 mem_sample_rate: float = -1.0):
        self.args = args
        self.mem_sample_rate = mem_sample_rate
        self.h5_file_path = h5_file_path
        self.train_part = train_part
        self.marker_num = marker_num
        self.nan_class_id = self.marker_num
        self.scale = args.DATA.scale
        self.use_window = self.args.DATA.use_window

        self.window_size = window_size
        with open(file_list_path, "r") as f:
            self.file_list = f.readlines()
        self.file_list = [item.split() for item in self.file_list]
        self.clip_list = []
        if mem_sample_rate < 0:
            sample_item_idx = range(len(self.file_list))
        elif mem_sample_rate == 1.0:
            sample_item_idx = range(len(self.file_list))
        else:
            sample_item_idx = np.random.choice(range(len(self.file_list)),
                                               int(len(self.file_list) * self.mem_sample_rate), replace=False)
        self.sample_file_list = [self.file_list[i] for i in sample_item_idx]

        h5_file_path = os.path.join(args.DATA.root_dir, args.DATA.h5_path)

        self.marker_pos = []
        self.marker_vis = []
        self.corr_error = []
        self.marker_error = []

        if self.args.DATA.train_part == 'full_body':
            marker_name_config_path = os.path.join(args.DATA.root_dir,
                                                   args.DATA.marker_name_config_path)
            self.args.DATA.marker_name_config = json.load(open(marker_name_config_path, "r"))
            self.args.DATA.body_marker_num = len(self.args.DATA.marker_name_config["body"][self.args.DATA.body_type])
            self.args.DATA.hand_marker_num = len(
                self.args.DATA.marker_name_config["finger"][self.args.DATA.finger_type]) // 2

        if self.args.DATA.body_type == "grab":
            with h5py.File(h5_file_path, "r") as h5_file:
                for item in tqdm(self.sample_file_list):
                    if train_part in ["body", "left_hand", "right_hand"]:
                        character_data = h5_file[item[0]][item[1]][self.train_part]
                        marker_pos_arr = np.asarray(character_data["fix_marker_pos"])[::2]
                        marker_vis_arr = np.asarray(character_data["marker_vis"])[::2]

                        if not self.args.DATA.global_transform_align:
                            global_trans = np.asarray(character_data["global_trans"])[::2]
                            global_rot_mat = np.asarray(character_data["global_rot_mat"])[::2]

                            transformed_marker_pos = copy.deepcopy(marker_pos_arr)
                            for i in range(marker_pos_arr.shape[0]):
                                transformed_marker_pos[i] = \
                                    add_marker_global_transform(marker_pos_arr[i], global_rot_mat[i], global_trans[i])
                            marker_pos_arr = transformed_marker_pos

                        marker_pos_arr_nan_pos = np.where(np.isnan(marker_pos_arr))
                        no_nan_pos_idx = np.setdiff1d(np.arange(marker_pos_arr.shape[0]), marker_pos_arr_nan_pos[0])

                        self.marker_pos.append(marker_pos_arr[no_nan_pos_idx])
                        self.marker_vis.append(marker_vis_arr[no_nan_pos_idx])
                    else:
                        body_data = h5_file[item[0]][item[1]]["body"]
                        body_marker_pos_arr = np.asarray(body_data["fix_marker_pos"])[::2]
                        body_marker_vis_arr = np.asarray(body_data["marker_vis"])[::2]

                        body_global_trans = np.asarray(body_data["global_trans"])[::2]
                        body_global_rot_mat = np.asarray(body_data["global_rot_mat"])[::2]
                        transformed_body_marker_pos = copy.deepcopy(body_marker_pos_arr)
                        for i in range(body_marker_pos_arr.shape[0]):
                            transformed_body_marker_pos[i] = \
                                add_marker_global_transform(body_marker_pos_arr[i], body_global_rot_mat[i],
                                                            body_global_trans[i])

                        left_hand_data = h5_file[item[0]][item[1]]["left_hand"]
                        left_hand_marker_pos_arr = np.asarray(left_hand_data["fix_marker_pos"])[::2]
                        left_hand_marker_vis_arr = np.asarray(left_hand_data["marker_vis"])[::2]

                        left_hand_global_trans = np.asarray(left_hand_data["global_trans"])[::2]
                        left_hand_global_rot_mat = np.asarray(left_hand_data["global_rot_mat"])[::2]
                        transformed_left_hand_marker_pos = copy.deepcopy(left_hand_marker_pos_arr)
                        for i in range(left_hand_marker_pos_arr.shape[0]):
                            transformed_left_hand_marker_pos[i] = \
                                add_marker_global_transform(left_hand_marker_pos_arr[i],
                                                            left_hand_global_rot_mat[i],
                                                            left_hand_global_trans[i])

                        right_hand_data = h5_file[item[0]][item[1]]["right_hand"]
                        right_hand_marker_pos_arr = np.asarray(right_hand_data["fix_marker_pos"])[::2]
                        right_hand_marker_vis_arr = np.asarray(right_hand_data["marker_vis"])[::2]

                        right_hand_global_trans = np.asarray(right_hand_data["global_trans"])[::2]
                        right_hand_global_rot_mat = np.asarray(right_hand_data["global_rot_mat"])[::2]
                        transformed_right_hand_marker_pos = copy.deepcopy(right_hand_marker_pos_arr)
                        for i in range(right_hand_marker_pos_arr.shape[0]):
                            transformed_right_hand_marker_pos[i] = \
                                add_marker_global_transform(right_hand_marker_pos_arr[i],
                                                            right_hand_global_rot_mat[i],
                                                            right_hand_global_trans[i])

                        transformed_full_body_marker_pos = \
                            np.concatenate([transformed_body_marker_pos,
                                            transformed_left_hand_marker_pos,
                                            transformed_right_hand_marker_pos], axis=2)
                        full_body_marker_vis_arr = \
                            np.concatenate([body_marker_vis_arr,
                                            left_hand_marker_vis_arr,
                                            right_hand_marker_vis_arr], axis=2)

                        if not self.args.DATA.global_transform_align:
                            marker_pos_arr = transformed_full_body_marker_pos
                            marker_vis_arr = full_body_marker_vis_arr

                        else:
                            marker_pos_arr = copy.deepcopy(transformed_full_body_marker_pos)
                            for i in range(transformed_full_body_marker_pos.shape[0]):
                                marker_pos_arr[i] = \
                                    remove_marker_global_transform(transformed_full_body_marker_pos[i],
                                                                   body_global_rot_mat[i],
                                                                   body_global_trans[i])

                            marker_vis_arr = full_body_marker_vis_arr

                        marker_pos_arr_nan_pos = np.where(np.isnan(transformed_full_body_marker_pos))
                        no_nan_pos_idx = np.setdiff1d(np.arange(transformed_full_body_marker_pos.shape[0]),
                                                      marker_pos_arr_nan_pos[0])

                        self.marker_pos.append(marker_pos_arr[no_nan_pos_idx])
                        self.marker_vis.append(marker_vis_arr[no_nan_pos_idx])

            self.marker_pos = np.concatenate(self.marker_pos, axis=0)
            # self.marker_pos = (self.marker_pos - self.marker_pos_mean) / self.marker_pos_std

            self.marker_vis = np.concatenate(self.marker_vis, axis=0)

            if not self.use_window:
                self.marker_pos = self.marker_pos.reshape((-1, *self.marker_pos.shape[2:]))
                self.marker_vis = self.marker_vis.reshape((-1, *self.marker_vis.shape[2:]))

            self.marker_pos = self.marker_pos * self.scale

            # np.save("before_align.npy", self.marker_pos)
            if self.args.DATA.median_align:
                if self.use_window:
                    self.marker_pos = self.marker_pos.reshape((-1, *self.marker_pos.shape[2:]))
                    offset = compute_offsets_np(self.marker_pos)
                    self.marker_pos = self.marker_pos - offset
                    self.marker_pos = self.marker_pos. \
                        reshape((-1, self.args.DATA.window_size, *self.marker_pos.shape[1:]))
                else:
                    offset = compute_offsets_np(self.marker_pos)
                    self.marker_pos = self.marker_pos - offset
            # np.save("after_align.npy", self.marker_pos)
        else:
            with h5py.File(h5_file_path, "r") as h5_file:
                for item in tqdm(self.sample_file_list):
                    if train_part in ["body", "left_hand", "right_hand"]:
                        character_data = h5_file[item[0]][item[1]][self.train_part]
                        marker_pos_arr = np.asarray(character_data["no_fix_marker_pos"])[::2]
                        marker_vis_arr = np.asarray(character_data["marker_vis"])[::2]
                        marker_error_arr = np.asarray(character_data["marker_error"])[::2]
                        corr_error_arr = np.asarray(character_data["corr_error"])[::2]

                        if not self.args.DATA.global_transform_align:
                            global_trans = np.asarray(character_data["global_trans"])[::2]
                            global_rot_mat = np.asarray(character_data["global_rot_mat"])[::2]

                            transformed_marker_pos = copy.deepcopy(marker_pos_arr)
                            for i in range(marker_pos_arr.shape[0]):
                                transformed_marker_pos[i] = \
                                    add_marker_global_transform(marker_pos_arr[i], global_rot_mat[i], global_trans[i])
                            marker_pos_arr = transformed_marker_pos

                        marker_pos_arr_nan_pos = np.where(np.isnan(marker_pos_arr))
                        no_nan_pos_idx = np.setdiff1d(np.arange(marker_pos_arr.shape[0]), marker_pos_arr_nan_pos[0])

                        self.marker_pos.append(marker_pos_arr[no_nan_pos_idx])
                        self.marker_vis.append(marker_vis_arr[no_nan_pos_idx])
                        self.corr_error.append(corr_error_arr[no_nan_pos_idx])
                        self.marker_error.append(marker_error_arr[no_nan_pos_idx])
                    else:
                        body_data = h5_file[item[0]][item[1]]["body"]
                        body_marker_pos_arr = np.asarray(body_data["fix_marker_pos"])[::2]
                        body_marker_vis_arr = np.asarray(body_data["marker_vis"])[::2]
                        body_marker_error_arr = np.asarray(body_data["marker_error"])[::2]
                        body_corr_error_arr = np.asarray(body_data["corr_error"])[::2]
                        body_global_trans = np.asarray(body_data["global_trans"])[::2]
                        body_global_rot_mat = np.asarray(body_data["global_rot_mat"])[::2]
                        transformed_body_marker_pos = copy.deepcopy(body_marker_pos_arr)
                        for i in range(body_marker_pos_arr.shape[0]):
                            transformed_body_marker_pos[i] = \
                                add_marker_global_transform(body_marker_pos_arr[i], body_global_rot_mat[i],
                                                            body_global_trans[i])

                        left_hand_data = h5_file[item[0]][item[1]]["left_hand"]
                        left_hand_marker_pos_arr = np.asarray(left_hand_data["fix_marker_pos"])[::2]
                        left_hand_marker_vis_arr = np.asarray(left_hand_data["marker_vis"])[::2]
                        left_hand_marker_error_arr = np.asarray(left_hand_data["marker_error"])[::2]
                        left_hand_corr_error_arr = np.asarray(left_hand_data["corr_error"])[::2]
                        left_hand_global_trans = np.asarray(left_hand_data["global_trans"])[::2]
                        left_hand_global_rot_mat = np.asarray(left_hand_data["global_rot_mat"])[::2]
                        transformed_left_hand_marker_pos = copy.deepcopy(left_hand_marker_pos_arr)
                        for i in range(left_hand_marker_pos_arr.shape[0]):
                            transformed_left_hand_marker_pos[i] = \
                                add_marker_global_transform(left_hand_marker_pos_arr[i],
                                                            left_hand_global_rot_mat[i],
                                                            left_hand_global_trans[i])

                        right_hand_data = h5_file[item[0]][item[1]]["right_hand"]
                        right_hand_marker_pos_arr = np.asarray(right_hand_data["fix_marker_pos"])[::2]
                        right_hand_marker_vis_arr = np.asarray(right_hand_data["marker_vis"])[::2]
                        right_hand_marker_error_arr = np.asarray(right_hand_data["marker_error"])[::2]
                        right_hand_corr_error_arr = np.asarray(right_hand_data["corr_error"])[::2]
                        right_hand_global_trans = np.asarray(right_hand_data["global_trans"])[::2]
                        right_hand_global_rot_mat = np.asarray(right_hand_data["global_rot_mat"])[::2]
                        transformed_right_hand_marker_pos = copy.deepcopy(right_hand_marker_pos_arr)
                        for i in range(right_hand_marker_pos_arr.shape[0]):
                            transformed_right_hand_marker_pos[i] = \
                                add_marker_global_transform(right_hand_marker_pos_arr[i],
                                                            right_hand_global_rot_mat[i],
                                                            right_hand_global_trans[i])

                        transformed_full_body_marker_pos = \
                            np.concatenate([transformed_body_marker_pos,
                                            transformed_left_hand_marker_pos,
                                            transformed_right_hand_marker_pos], axis=2)
                        full_body_marker_vis_arr = \
                            np.concatenate([body_marker_vis_arr,
                                            left_hand_marker_vis_arr,
                                            right_hand_marker_vis_arr], axis=2)
                        full_body_marker_error_arr = \
                            np.concatenate([body_marker_error_arr,
                                            left_hand_marker_error_arr,
                                            right_hand_marker_error_arr],
                                           axis=2)
                        full_body_corr_error_arr = \
                            np.concatenate([body_corr_error_arr,
                                            left_hand_corr_error_arr,
                                            right_hand_corr_error_arr],
                                           axis=2)

                        if not self.args.DATA.global_transform_align:
                            marker_pos_arr = transformed_full_body_marker_pos
                            marker_vis_arr = full_body_marker_vis_arr
                            corr_error_arr = full_body_corr_error_arr
                            marker_error_arr = full_body_marker_error_arr
                        else:
                            marker_pos_arr = copy.deepcopy(transformed_full_body_marker_pos)
                            for i in range(transformed_full_body_marker_pos.shape[0]):
                                marker_pos_arr[i] = \
                                    remove_marker_global_transform(transformed_full_body_marker_pos[i],
                                                                   body_global_rot_mat[i],
                                                                   body_global_trans[i])

                            marker_vis_arr = full_body_marker_vis_arr
                            corr_error_arr = full_body_corr_error_arr
                            marker_error_arr = full_body_marker_error_arr

                        marker_pos_arr_nan_pos = np.where(np.isnan(transformed_full_body_marker_pos))
                        no_nan_pos_idx = np.setdiff1d(np.arange(transformed_full_body_marker_pos.shape[0]),
                                                      marker_pos_arr_nan_pos[0])

                        self.marker_pos.append(marker_pos_arr[no_nan_pos_idx])
                        self.marker_vis.append(marker_vis_arr[no_nan_pos_idx])
                        self.corr_error.append(corr_error_arr[no_nan_pos_idx])
                        self.marker_error.append(marker_error_arr[no_nan_pos_idx])

            self.marker_pos = np.concatenate(self.marker_pos, axis=0)
            # self.marker_pos = (self.marker_pos - self.marker_pos_mean) / self.marker_pos_std

            self.marker_vis = np.concatenate(self.marker_vis, axis=0)
            self.marker_error = np.concatenate(self.marker_error, axis=0)
            self.corr_error = np.concatenate(self.corr_error, axis=0)

            if not self.use_window:
                self.marker_pos = self.marker_pos.reshape((-1, *self.marker_pos.shape[2:]))
                self.marker_vis = self.marker_vis.reshape((-1, *self.marker_vis.shape[2:]))
                self.marker_error = self.marker_error.reshape((-1, *self.marker_error.shape[2:]))
                self.corr_error = self.corr_error.reshape((-1, *self.corr_error.shape[2:]))

            self.marker_pos = self.marker_pos * self.scale

            # np.save("before_align.npy", self.marker_pos)
            if self.args.DATA.median_align:
                if self.use_window:
                    self.marker_pos = self.marker_pos.reshape((-1, *self.marker_pos.shape[2:]))
                    offset = compute_offsets_np(self.marker_pos)
                    self.marker_pos = self.marker_pos - offset
                    self.marker_pos = self.marker_pos. \
                        reshape((-1, self.args.DATA.window_size, *self.marker_pos.shape[1:]))
                else:
                    offset = compute_offsets_np(self.marker_pos)
                    self.marker_pos = self.marker_pos - offset
            # np.save("after_align.npy", self.marker_pos)

    def __len__(self) -> int:
        return self.marker_pos.shape[0]

    def open_h5(self) -> None:
        self.h5_file = h5py.File(self.h5_file_path, "r")

    def __getitem__(self, item: int) -> tuple:
        if not hasattr(self, "h5_file"):
            self.open_h5()

        if self.args.DATA.body_type == "grab":
            marker_pos = self.marker_pos[item]
            marker_vis = self.marker_vis[item]
            corr_error = np.asarray([-1] * self.marker_num)
            marker_error = np.asarray([False] * self.marker_num)

            occlusion_marker_idx = np.random.choice(self.marker_num,
                                                    int(self.marker_num * self.args.DATA.occlusion_rate),
                                                    replace=False)
            marker_pos[occlusion_marker_idx] *= 0
            marker_vis[occlusion_marker_idx] = 1

            def prepare_data(marker_pos, marker_vis, corr_error, marker_error):
                perm_vec = np.random.permutation(marker_pos.shape[0])

                label_ids = np.array(list(range(self.marker_num)))
                markers_not_available = np.bitwise_or((np.sum(marker_pos == 0.0, axis=-1) == 3), marker_vis == 1)
                markers_not_available = np.bitwise_or(markers_not_available, marker_error)

                label_ids[markers_not_available] = self.nan_class_id

                if any(corr_error != -1):
                    label_ids[corr_error != -1] = corr_error[corr_error != -1]
                labels_not_available = copy.deepcopy(markers_not_available).astype(np.float32)

                label_ids[np.where(np.sum(marker_pos == 0.0, axis=-1) == 3)[0]] = self.nan_class_id

                shuffled_marker_pos = marker_pos[perm_vec]
                shuffled_label_ids = label_ids[perm_vec]
                shuffled_label_one_hots = torch.nn.functional.one_hot(torch.from_numpy(shuffled_label_ids),
                                                                      num_classes=self.marker_num + 1).numpy()
                if self.args.EVAL.export_vicon:
                    vicon_res = np.array(list(range(self.marker_num)))
                    markers_not_available = np.bitwise_or((np.sum(marker_pos == 0.0, axis=-1) == 3), marker_vis == 1)
                    vicon_res[markers_not_available] = self.nan_class_id
                    shuffled_vicon_res = vicon_res[perm_vec]

                nan_mask = (shuffled_label_ids == self.nan_class_id)
                label_weights = np.ones_like(label_ids, dtype=np.float32)

                label_nan_weight = np.divide(1.0, np.sum(nan_mask) if np.sum(nan_mask) != 0 else 1.0)
                label_weights[label_ids == self.nan_class_id] = label_nan_weight

                label_dustbin = \
                    np.concatenate([labels_not_available,
                                    np.asarray([self.marker_num + 1 -
                                                labels_not_available.sum().astype(np.float32)])])

                aug_asmat = np.concatenate([shuffled_label_one_hots, label_dustbin.reshape(1, -1)], axis=0).astype(
                    np.float32)
                aug_asmat_weights = \
                    np.concatenate([label_weights.reshape((-1, 1)).repeat(self.marker_num + 1, -1),
                                    np.ones_like(label_dustbin).reshape((1, -1)) * label_nan_weight], axis=0).astype(
                        np.float32)

                out_dict = {"perm_vec": perm_vec,
                            "marker_pos": shuffled_marker_pos,
                            "label_ids": shuffled_label_ids,
                            "aug_asmat": aug_asmat,
                            "aug_asmat_weights": aug_asmat_weights}

                if self.args.EVAL.export_vicon:
                    out_dict["vicon_ids"] = shuffled_vicon_res

                if self.train_part == "full_body":
                    part_id = np.zeros(self.marker_num).astype(int)
                    # part_id[:self.bo]
                    part_id[:self.args.DATA.body_marker_num] = 0
                    part_id[self.args.DATA.body_marker_num: \
                            self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num] = 1
                    part_id[self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num: \
                            self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num * 2] = 2
                    part_one_hot = torch.nn.functional.one_hot(torch.from_numpy(part_id), num_classes=3).numpy()

                    shuffled_part_id = part_id[perm_vec]
                    shuffled_part_one_hot = part_one_hot[perm_vec]

                    out_dict.update({"part_ids": shuffled_part_id,
                                     "part_one_hot": shuffled_part_one_hot})
                return out_dict

            if self.use_window:
                out_dict_list = []
                for i in range(self.window_size):
                    out_dict_list.append(prepare_data(marker_pos[i], marker_vis[i], corr_error[i], marker_error[i]))
                out_dict = {}
                for key in out_dict_list[0]:
                    out_dict[key] = np.stack([d[key] for d in out_dict_list])
            else:
                out_dict = prepare_data(marker_pos, marker_vis, corr_error, marker_error)

            return out_dict
        else:
            marker_pos = self.marker_pos[item]
            marker_vis = self.marker_vis[item]
            corr_error = self.corr_error[item]
            marker_error = self.marker_error[item]

            def prepare_data(marker_pos, marker_vis, corr_error, marker_error):
                perm_vec = np.random.permutation(marker_pos.shape[0])

                label_ids = np.array(list(range(self.marker_num)))
                markers_not_available = np.bitwise_or((np.sum(marker_pos == 0.0, axis=-1) == 3), marker_vis == 1)
                markers_not_available = np.bitwise_or(markers_not_available, marker_error)

                label_ids[markers_not_available] = self.nan_class_id

                if any(corr_error != -1):
                    label_ids[corr_error != -1] = corr_error[corr_error != -1]
                labels_not_available = copy.deepcopy(markers_not_available).astype(np.float32)

                label_ids[np.where(np.sum(marker_pos == 0.0, axis=-1) == 3)[0]] = self.nan_class_id

                shuffled_marker_pos = marker_pos[perm_vec]
                shuffled_label_ids = label_ids[perm_vec]
                shuffled_label_one_hots = torch.nn.functional.one_hot(torch.from_numpy(shuffled_label_ids),
                                                                      num_classes=self.marker_num + 1).numpy()
                if self.args.EVAL.export_vicon:
                    vicon_res = np.array(list(range(self.marker_num)))
                    markers_not_available = np.bitwise_or((np.sum(marker_pos == 0.0, axis=-1) == 3), marker_vis == 1)
                    vicon_res[markers_not_available] = self.nan_class_id
                    shuffled_vicon_res = vicon_res[perm_vec]

                nan_mask = (shuffled_label_ids == self.nan_class_id)
                label_weights = np.ones_like(label_ids, dtype=np.float32)

                label_nan_weight = np.divide(1.0, np.sum(nan_mask) if np.sum(nan_mask) != 0 else 1.0)
                label_weights[label_ids == self.nan_class_id] = label_nan_weight

                label_dustbin = \
                    np.concatenate([labels_not_available,
                                    np.asarray([self.marker_num + 1 -
                                                labels_not_available.sum().astype(np.float32)])])

                aug_asmat = np.concatenate([shuffled_label_one_hots, label_dustbin.reshape(1, -1)], axis=0).astype(
                    np.float32)
                aug_asmat_weights = \
                    np.concatenate([label_weights.reshape((-1, 1)).repeat(self.marker_num + 1, -1),
                                    np.ones_like(label_dustbin).reshape((1, -1)) * label_nan_weight], axis=0).astype(
                        np.float32)

                out_dict = {'perm_vec': perm_vec,
                            "marker_pos": shuffled_marker_pos,
                            "label_ids": shuffled_label_ids,
                            "aug_asmat": aug_asmat,
                            "aug_asmat_weights": aug_asmat_weights}

                if self.args.EVAL.export_vicon:
                    out_dict["vicon_ids"] = shuffled_vicon_res

                if self.train_part == "full_body":
                    part_id = np.zeros(self.marker_num).astype(int)
                    # part_id[:self.bo]
                    part_id[:self.args.DATA.body_marker_num] = 0
                    part_id[self.args.DATA.body_marker_num: \
                            self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num] = 1
                    part_id[self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num: \
                            self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num * 2] = 2
                    part_one_hot = torch.nn.functional.one_hot(torch.from_numpy(part_id), num_classes=3).numpy()

                    shuffled_part_id = part_id[perm_vec]
                    shuffled_part_one_hot = part_one_hot[perm_vec]

                    out_dict.update({"part_ids": shuffled_part_id,
                                     "part_one_hot": shuffled_part_one_hot})
                return out_dict

        if self.use_window:
            out_dict_list = []
            for i in range(self.window_size):
                out_dict_list.append(prepare_data(marker_pos[i], marker_vis[i], corr_error[i], marker_error[i]))
            out_dict = {}
            for key in out_dict_list[0]:
                out_dict[key] = np.stack([d[key] for d in out_dict_list])
        else:
            out_dict = prepare_data(marker_pos, marker_vis, corr_error, marker_error)

        return out_dict


class MarkerLabelDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        marker_name_config_path = os.path.join(args.DATA.root_dir,
                                               args.DATA.marker_name_config_path)
        marker_name_config = json.load(open(marker_name_config_path, "r"))

        train_part = args.DATA.train_part

        if train_part in ["left_hand", "right_hand"]:
            marker_name_list = marker_name_config["finger"][args.DATA.finger_type]
            marker_num = len(marker_name_list) // 2
        elif train_part == "body":
            marker_name_list = marker_name_config["body"][args.DATA.body_type]
            marker_num = len(marker_name_list)
        elif train_part == "full_body":
            marker_list = marker_name_config["body"][args.DATA.body_type] + \
                          marker_name_config["finger"][args.DATA.finger_type]
            marker_num = len(marker_list)
        else:
            raise Exception("Wrong training part")

        h5_file_path = os.path.join(args.DATA.root_dir, args.DATA.h5_path)
        train_list_path = os.path.join(args.DATA.root_dir, args.DATA.train_list_path)
        test_list_path = os.path.join(args.DATA.root_dir, args.DATA.test_list_path)

        if args.EVAL.only_eval:
            self.train_data_loader = None
        else:
            self.train_data_loader = DataLoader(
                MarkerLabelDataset(args, h5_file_path, train_list_path, marker_num,
                                   args.DATA.window_size, args.DATA.train_part,
                                   mem_sample_rate=args.DATA.train_mem_sample_rate),
                batch_size=args.TRAIN.batch_size,
                num_workers=args.DATA.num_workers,
                # shuffle=False,
                shuffle=True,
                pin_memory=True)

        self.test_data_loader = DataLoader(
            MarkerLabelDataset(args, h5_file_path, test_list_path, marker_num,
                               args.DATA.window_size, args.DATA.train_part,
                               mem_sample_rate=args.DATA.test_mem_sample_rate),
            batch_size=args.TRAIN.batch_size, num_workers=args.DATA.num_workers,
            shuffle=False, pin_memory=True)

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.test_data_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_data_loader
