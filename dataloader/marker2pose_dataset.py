import os

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from tqdm import tqdm
import h5py
import json

import numpy as np
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningDataModule


class Marker2PoseDataset(Dataset):
    def __init__(self, args, h5_file_path: str, file_list_path: str, marker_num: int,
                 window_size: int, train_part: str,
                 mem_sample_rate: float = -1.0):
        self.args = args
        self.rot_repr = args.DATA.rot_repr
        self.mem_sample_rate = mem_sample_rate
        self.h5_file_path = h5_file_path
        self.train_part = train_part
        self.marker_num = marker_num
        self.window_size = window_size
        self.use_window = args.DATA.use_window
        with open(file_list_path, "r") as f:
            self.file_list = f.readlines()
        # self.file_list = self.file_list[:100]
        self.file_list = [item.split() for item in self.file_list]
        self.clip_list = []
        if mem_sample_rate < 0 or mem_sample_rate == 1.0:
            sample_item_idx = range(len(self.file_list))
        else:
            sample_item_idx = np.random.choice(range(len(self.file_list)),
                                               int(len(self.file_list) * self.mem_sample_rate), replace=False)
        self.sample_file_list = [self.file_list[i] for i in sample_item_idx]

        h5_file_path = os.path.join(args.DATA.root_dir, args.DATA.h5_path)
        marker_pos_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.marker_pos_mean_std_path)
        joint_pos_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.joint_pos_mean_std_path)
        motion_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.motion_mean_std_path)
        motion_mat_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.motion_mat_mean_std_path)
        motion_trans_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.motion_trans_mean_std_path)
        joint_offset_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.joint_offset_mean_std_path)

        cos_twist_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.cos_twist_mean_std_path)
        sin_twist_mean_std_path = os.path.join(args.DATA.root_dir, args.DATA.sin_twist_mean_std_path)

        if self.rot_repr == "quat":
            motion_mean_std = np.load(motion_mean_std_path)
            self.motion_mean = motion_mean_std["mean"]
            self.motion_std = motion_mean_std["std"]
            self.motion_mean = self.motion_mean.reshape((1, self.motion_mean.shape[0] * 3 * 3)). \
                astype(np.float32)
            self.motion_std = self.motion_std.reshape(self.motion_mean.shape).astype(np.float32)
        elif self.rot_repr == "joint_pos":
            joint_pos_mean_std = np.load(joint_pos_mean_std_path)
            self.joint_pos_mean = joint_pos_mean_std["mean"]
            self.joint_pos_std = joint_pos_mean_std["std"]
            self.joint_pos_mean = self.joint_pos_mean.reshape((1, self.joint_pos_mean.shape[0], 3)). \
                astype(np.float32)
            self.joint_pos_std = self.joint_pos_std.reshape(self.joint_pos_mean.shape).astype(np.float32)

            cos_twist_mean_std = np.load(cos_twist_mean_std_path)
            self.cos_twist_mean = cos_twist_mean_std["mean"]
            self.cos_twist_std = cos_twist_mean_std["std"]
            self.cos_twist_mean = self.cos_twist_mean.reshape((1, -1)).astype(np.float32)
            self.cos_twist_std = self.cos_twist_std.reshape(self.cos_twist_mean.shape).astype(np.float32)

            sin_twist_mean_std = np.load(sin_twist_mean_std_path)
            self.sin_twist_mean = sin_twist_mean_std["mean"]
            self.sin_twist_std = sin_twist_mean_std["std"]
            self.sin_twist_mean = self.sin_twist_mean.reshape((1, -1)).astype(np.float32)
            self.sin_twist_std = self.sin_twist_std.reshape(self.sin_twist_mean.shape).astype(np.float32)

            self.phi_std = np.concatenate([np.expand_dims(self.cos_twist_std, axis=-1),
                                           np.expand_dims(self.sin_twist_std, axis=-1)], -1)
            self.phi_mean = np.concatenate([np.expand_dims(self.cos_twist_mean, axis=-1),
                                            np.expand_dims(self.sin_twist_mean, axis=-1)], -1)
        elif self.rot_repr == "mat_full":
            pass
        else:
            raise NotImplementedError

        marker_pos_mean_std = np.load(marker_pos_mean_std_path)
        self.marker_pos_mean = marker_pos_mean_std["mean"]
        self.marker_pos_std = marker_pos_mean_std["std"]
        self.marker_pos_mean = self.marker_pos_mean. \
            reshape((1, self.marker_pos_mean.shape[0], self.marker_pos_mean.shape[1])).astype(np.float32)
        self.marker_pos_std = self.marker_pos_std.reshape(self.marker_pos_mean.shape).astype(np.float32)

        joint_offset_mean_std = np.load(joint_offset_mean_std_path)
        self.joint_offset_mean = joint_offset_mean_std["mean"].astype(np.float32)
        self.joint_offset_std = joint_offset_mean_std["std"].astype(np.float32)
        self.joint_offset_mean = self.joint_offset_mean.reshape(
            (1, self.joint_offset_mean.shape[0], self.joint_offset_mean.shape[1]))
        self.joint_offset_std = self.joint_offset_std.reshape(self.joint_offset_mean.shape)

        motion_mat_mean_std = np.load(motion_mat_mean_std_path)
        self.motion_mat_mean = motion_mat_mean_std["mean"]
        self.motion_mat_std = motion_mat_mean_std["std"]
        self.motion_mat_mean = self.motion_mat_mean.reshape((1, self.motion_mat_mean.shape[0] * 3 * 3)). \
            astype(np.float32)
        self.motion_mat_std = self.motion_mat_std.reshape(self.motion_mat_mean.shape).astype(np.float32)

        motion_trans_mean_std = np.load(motion_trans_mean_std_path)
        self.motion_trans_mean = motion_trans_mean_std["mean"]
        self.motion_trans_std = motion_trans_mean_std["std"]
        self.motion_trans_mean = self.motion_trans_mean.reshape((1, self.motion_trans_mean.shape[0])).astype(
            np.float32)
        self.motion_trans_std = self.motion_trans_std.reshape(self.motion_trans_mean.shape).astype(np.float32)

        self.marker_pos = []
        self.motion = []
        self.marker_vis = []
        self.joint_offset = []
        self.joint_pos = []
        self.phi = []
        self.name_list = []
        self.global_trans = []
        self.global_rot = []

        if args.DATA.use_wrist:
            wrist_marker_pos_mean_std = np.load(
                os.path.join(args.DATA.root_dir, args.DATA.wrist_marker_pos_mean_std_path))
            wrist_marker_pos_mean = wrist_marker_pos_mean_std["mean"]
            wrist_marker_pos_std = wrist_marker_pos_mean_std["std"]
            self.wrist_marker_pos_mean = wrist_marker_pos_mean.reshape((1, *wrist_marker_pos_mean.shape))
            self.wrist_marker_pos_std = wrist_marker_pos_std.reshape(wrist_marker_pos_mean.shape)

            self.wrist_marker_pos = []

        with h5py.File(h5_file_path, "r") as h5_file:
            for item in tqdm(self.sample_file_list):
                character_data = h5_file[item[0]][item[1]][self.train_part]
                # marker_pos_arr = np.asarray(character_data["fix_occlusion_marker_pos"])
                marker_pos_arr = np.asarray(character_data["fix_marker_pos"])
                joint_pos_arr = np.asarray(character_data["joint_pos"])

                global_trans_arr = np.asarray(character_data["global_trans"])
                global_rot_arr = np.asarray(character_data["global_rot_mat"])
                if self.rot_repr == "quat":
                    motion_arr = np.asarray(character_data["motion_quat_trans"])
                elif self.rot_repr == "mat_full":
                    motion_arr = np.concatenate([np.asarray(character_data["motion_mat"]). \
                                                reshape((character_data["motion_mat"].shape[0],
                                                         character_data["motion_mat"].shape[1], -1)),
                                                 np.asarray(character_data["motion_trans"])], axis=2)
                elif self.rot_repr == "joint_pos":
                    motion_arr = np.concatenate([np.asarray(character_data["motion_mat"]). \
                                                reshape((character_data["motion_mat"].shape[0],
                                                         character_data["motion_mat"].shape[1], -1)),
                                                 np.asarray(character_data["motion_trans"])], axis=2)
                else:
                    raise NotImplementedError
                marker_vis_arr = np.asarray(character_data["marker_vis"])

                marker_pos_arr_nan_pos = np.where(np.isnan(marker_pos_arr))
                motion_nan_pos = np.where(np.isnan(motion_arr))
                no_nan_pos_idx = np.setdiff1d(np.arange(marker_pos_arr.shape[0]), marker_pos_arr_nan_pos[0])
                no_nan_pos_idx = np.setdiff1d(no_nan_pos_idx, motion_nan_pos)
                if args.DATA.use_wrist:
                    wrist_marker_pos_arr = np.asarray(character_data["wrist_marker_pos"])
                    wrist_arr_nan_pos = np.where(np.isnan(wrist_marker_pos_arr))
                    no_nan_pos_idx = np.setdiff1d(no_nan_pos_idx, wrist_arr_nan_pos)

                # print(sum([arr.shape[0] for arr in self.marker_pos]))
                item_window_num = no_nan_pos_idx.shape[0]
                train_part_joint_pos = np.asarray(character_data["t_pose"])
                train_part_parent_joint_pos = train_part_joint_pos[args.DATA.parent_list]
                joint_offset = train_part_joint_pos - train_part_parent_joint_pos
                joint_offset[0] *= 0
                joint_offset = joint_offset.astype(np.float32)
                self.joint_offset.append(np.repeat(np.expand_dims(joint_offset, [0]), item_window_num, axis=0))

                # swing_mat = np.asarray(character_data["swing_mat"])
                # twist_mat = np.asarray(character_data["twist_mat"])

                self.marker_pos.append(marker_pos_arr[no_nan_pos_idx])
                self.motion.append(motion_arr[no_nan_pos_idx])
                self.marker_vis.append(marker_vis_arr[no_nan_pos_idx])
                self.joint_pos.append(joint_pos_arr[no_nan_pos_idx])

                self.name_list.extend([item] * len(no_nan_pos_idx))
                self.global_rot.append(global_rot_arr[no_nan_pos_idx])
                self.global_trans.append(global_trans_arr[no_nan_pos_idx])
                # self.swing_mat.append(swing_mat[no_nan_pos_idx])
                # self.twist_mat.append(twist_mat[no_nan_pos_idx])

                if self.rot_repr == "joint_pos":
                    cos_twist_arr = np.asarray(character_data["cos_twist"])
                    sin_twist_arr = np.asarray(character_data["sin_twist"])
                    phi = np.concatenate([np.expand_dims(cos_twist_arr, -1),
                                          np.expand_dims(sin_twist_arr, -1)], axis=-1).astype(np.float32)
                    self.phi.append(phi[no_nan_pos_idx])

                if args.DATA.use_wrist:
                    self.wrist_marker_pos.append(wrist_marker_pos_arr[no_nan_pos_idx])

        self.motion = np.concatenate(self.motion, axis=0)
        # self.motion = (self.motion - self.motion_mean) / self.motion_std
        self.joint_pos = np.concatenate(self.joint_pos, axis=0)

        self.marker_pos = np.concatenate(self.marker_pos, axis=0)

        self.marker_pos += np.random.randn(*self.marker_pos.shape) * self.args.EVAL.jitter_intensity

        self.marker_pos = (self.marker_pos - self.marker_pos_mean) / self.marker_pos_std

        self.marker_vis = np.concatenate(self.marker_vis, axis=0)
        self.joint_offset = np.concatenate(self.joint_offset, axis=0)
        if self.rot_repr == "joint_pos":
            self.phi = np.concatenate(self.phi, axis=0)

        self.global_rot = np.concatenate(self.global_rot, axis=0)
        self.global_trans = np.concatenate(self.global_trans, axis=0)

        if not self.use_window:
            self.motion = self.motion.reshape((-1, *self.motion.shape[2:]))
            self.joint_pos = self.joint_pos.reshape((-1, *self.joint_pos.shape[2:]))
            self.marker_pos = self.marker_pos.reshape((-1, *self.marker_pos.shape[2:]))
            self.marker_vis = self.marker_vis.reshape((-1, *self.marker_vis.shape[2:]))
            self.phi = self.phi.reshape((-1, *self.phi.shape[2:]))

            self.joint_offset = np.repeat(np.expand_dims(self.joint_offset, 1), self.window_size, 1). \
                reshape((-1, *self.joint_offset.shape[1:]))

        if args.DATA.use_wrist:
            self.wrist_marker_pos = np.concatenate(self.wrist_marker_pos, axis=0)
            self.wrist_marker_pos = (self.wrist_marker_pos - self.wrist_marker_pos_mean) / \
                                    self.wrist_marker_pos_std

    def __len__(self) -> int:
        if self.args.EVAL.export_res:
            return len(self.sample_file_list)
        else:
            return self.marker_pos.shape[0]

    def open_h5(self) -> None:
        self.h5_file = h5py.File(self.h5_file_path, "r")

    def __getitem__(self, item: int) -> tuple:
        if not hasattr(self, "h5_file"):
            self.open_h5()

        if self.args.EVAL.export_res:
            name = self.sample_file_list[item]
            name_corr_idx = [n for n in range(len(self.name_list)) if self.name_list[n] == name]

            marker_pos = self.marker_pos[name_corr_idx]
            motion = self.motion[name_corr_idx]
            marker_vis = self.marker_vis[name_corr_idx]
            joint_offset = self.joint_offset[name_corr_idx]
            joint_pos = self.joint_pos[name_corr_idx]
            phi = self.phi[name_corr_idx]
            global_rot_mat = self.global_rot[name_corr_idx]
            global_trans = self.global_trans[name_corr_idx]

            if self.args.DATA.use_wrist:
                wrist_marker_pos = self.wrist_marker_pos[name_corr_idx]

        else:
            marker_pos = self.marker_pos[item]
            motion = self.motion[item]
            marker_vis = self.marker_vis[item]
            joint_offset = self.joint_offset[item]
            joint_pos = self.joint_pos[item]
            if self.rot_repr == "joint_pos":
                phi = self.phi[item]
            name = self.name_list[item]
            global_rot_mat = self.global_rot[item]
            global_trans = self.global_trans[item]

            if self.args.DATA.use_wrist:
                wrist_marker_pos = self.wrist_marker_pos[item]

        if self.rot_repr == "joint_pos":
            if self.args.DATA.use_wrist:
                return (marker_pos, marker_vis, wrist_marker_pos, joint_offset,
                        motion, joint_pos, phi, name, global_rot_mat, global_trans)
            else:
                return marker_pos, marker_vis, joint_offset, motion, joint_pos, phi, name, global_rot_mat, global_trans
        else:
            return marker_pos, marker_vis, joint_offset, motion, joint_pos, name, global_rot_mat, global_trans


class Marker2PoseDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

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
        else:
            raise Exception("Wrong training part")

        h5_file_path = os.path.join(args.DATA.root_dir, args.DATA.h5_path)
        train_list_path = os.path.join(args.DATA.root_dir, args.DATA.train_list_path)
        test_list_path = os.path.join(args.DATA.root_dir, args.DATA.test_list_path)

        if args.EVAL.only_eval:
            self.train_data_loader = None
        else:
            self.train_data_loader = DataLoader(
                Marker2PoseDataset(args, h5_file_path, train_list_path, marker_num,
                                   args.DATA.window_size, args.DATA.train_part,
                                   mem_sample_rate=args.DATA.train_mem_sample_rate),
                batch_size=args.TRAIN.batch_size, shuffle=True, pin_memory=True)

        self.test_data_loader = DataLoader(Marker2PoseDataset(args, h5_file_path, test_list_path, marker_num,
                                                              args.DATA.window_size, args.DATA.train_part,
                                                              mem_sample_rate=args.DATA.test_mem_sample_rate),
                                           batch_size=args.TRAIN.batch_size if not args.EVAL.export_res else 1,
                                           shuffle=False, pin_memory=True)

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.test_data_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_data_loader
