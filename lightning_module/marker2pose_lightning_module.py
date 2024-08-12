import copy

import numpy as np
import torch
import os
import argparse

from pytorch_lightning import LightningModule
from utils.utils import AverageMeter

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import Adam, RMSprop
from utils.utils import *
from utils.generate_bvh import export_to_bvh
from network.ik import *


class Marker2PoseLightningModule(LightningModule):
    def __init__(self, args, network, data_module):
        super().__init__()
        self.args = args
        self.use_window = args.DATA.use_window
        self.rot_repr = args.DATA.rot_repr
        self.network = network

        self.log_interval = args.TRAIN.log_interval
        self.save_interval = args.TRAIN.save_interval
        self.resume_checkpoint = args.TRAIN.resume_checkpoint

        self.use_wrist = args.DATA.use_wrist

        dataset = data_module.test_data_loader.dataset
        self.motion_mat_mean = torch.from_numpy(dataset.motion_mat_mean)
        self.motion_mat_std = torch.from_numpy(dataset.motion_mat_std)
        self.motion_trans_mean = torch.from_numpy(dataset.motion_trans_mean)
        self.motion_trans_std = torch.from_numpy(dataset.motion_trans_std)
        self.marker_pos_mean = torch.from_numpy(dataset.marker_pos_mean)
        self.marker_pos_std = torch.from_numpy(dataset.marker_pos_std)
        self.joint_offset_mean = torch.from_numpy(dataset.joint_offset_mean)
        self.joint_offset_std = torch.from_numpy(dataset.joint_offset_std)

        self.motion_mean = torch.cat([self.motion_mat_mean, self.motion_trans_mean], dim=1)
        self.motion_std = torch.cat([self.motion_mat_std, self.motion_trans_std], dim=1)

        if self.rot_repr == "joint_pos":
            self.joint_pos_mean = torch.from_numpy(dataset.joint_pos_mean)
            self.joint_pos_std = torch.from_numpy(dataset.joint_pos_std)

            self.phi_mean = torch.from_numpy(dataset.phi_mean)
            self.phi_std = torch.from_numpy(dataset.phi_std)

        self.test_result_list = []

    def configure_optimizers(self):
        optimizer_args = self.args.TRAIN.optimizer
        lr_scheduler_args = self.args.TRAIN.lr_scheduler

        optimize_params = [a[1] for a in self.network.named_parameters() if a[1].requires_grad]
        if optimizer_args.type == "Adam":
            optimizer = Adam(optimize_params, **optimizer_args.args)
        elif optimizer_args.type == "RMSprop":
            optimizer = RMSprop(optimize_params, **optimizer_args.args)
        else:
            raise NotImplementedError

        if lr_scheduler_args.type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_args.args)
        elif lr_scheduler_args.type == "StepLR":
            scheduler = StepLR(optimizer, **lr_scheduler_args.args)
        else:
            raise NotImplementedError

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_total_loss",
                    "interval": "epoch",
                    "frequency": 1}}

    def init_loss_meter(self, train_or_val):
        meters = {
            "total_loss": AverageMeter(),
            "motion_mean_squared_error": AverageMeter(),
            "mean_angle_error": AverageMeter(),
            "trans_error": AverageMeter(),
            "occ_mean_squared_error": AverageMeter(),
            "occ_mean_angle_error": AverageMeter(),
            "occ_trans_error": AverageMeter(),
            "fk_joint_pos_error": AverageMeter(),
            "estimate_joint_pos_error": AverageMeter(),
            "joint_offset_error": AverageMeter(),
            "phi_error": AverageMeter()}
        if train_or_val == "train":
            self.train_meters = copy.deepcopy(meters)
        elif train_or_val in ["val", "test"]:
            self.test_meters = copy.deepcopy(meters)
        else:
            raise Exception()

    def update_loss(self, loss_dict, train_or_val):
        if train_or_val == "train":
            meters = self.train_meters
        elif train_or_val in ["val", "test"]:
            meters = self.test_meters
        else:
            raise Exception()

        for item in loss_dict:
            if item in meters:
                meters[item].update(loss_dict[item].item())

    def log_loss(self, train_or_val):
        if train_or_val == "train":
            meters = self.train_meters
        elif train_or_val in ["val", "test"]:
            meters = self.test_meters
        else:
            raise Exception()
        self.log("step", self.current_epoch)
        self.logger.experiment.add_scalars(train_or_val, dict([(key, meters[key].avg) for key in meters]),
                                           self.current_epoch)

        for item in meters:
            self.log("{}_{}".format(train_or_val, item), meters[item].avg,
                     on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss_dict = self.model_forward(batch)
        self.update_loss(loss_dict, "train")

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.model_forward(batch)
        self.update_loss(loss_dict, "val")

    def test_step(self, batch, batch_idx):
        loss_dict = self.model_forward(batch)
        self.update_loss(loss_dict, "val")

        res_dict = {"name": loss_dict["name"],
                    "gt_motion": loss_dict["gt_motion"],
                    "pred_motion": loss_dict["pred_motion"],
                    "gt_joint_offset": loss_dict["gt_joint_offset"],
                    "pred_joint_offset": loss_dict["pred_joint_offset"],
                    "global_trans": loss_dict["global_trans"],
                    "global_rot_mat": loss_dict["global_rot_mat"],
                    "mean_angle_error": loss_dict["mean_angle_error"].item()}

        self.test_result_list.append(res_dict)

    def on_train_epoch_start(self) -> None:
        self.init_loss_meter("train")

    def on_validation_epoch_start(self) -> None:
        self.init_loss_meter("val")

    def on_test_epoch_start(self) -> None:
        self.init_loss_meter("test")

    def on_train_epoch_end(self) -> None:
        self.log_loss("train")

    def on_validation_epoch_end(self) -> None:
        self.log_loss("val")

    def on_test_epoch_end(self) -> None:
        def construct_transform_matrix(rot_mat_arr, trans_arr):
            # rotation -> translation
            assert rot_mat_arr.shape[0] == trans_arr.shape[0]
            assert rot_mat_arr.shape[1] == 3 and rot_mat_arr.shape[2] == 3 and trans_arr.shape[1] == 3
            anim_length = rot_mat_arr.shape[0]
            eye_mat_arr = np.repeat(np.eye(4).reshape((1, 4, 4)), anim_length, axis=0)
            trans = copy.deepcopy(eye_mat_arr)
            rot = copy.deepcopy(eye_mat_arr)
            rot[:, :3, :3] = rot_mat_arr
            trans[:, :3, 3] = trans_arr

            return trans @ rot

        def calc_joint_pos_trans(topology, motion, joint_offset, rot_repr="mat_full"):
            batch_size = motion.shape[0]
            trans = motion[..., -3:]
            if rot_repr == "quat":
                rot_quat = motion[..., :-3].reshape((batch_size, -1, 4))
                joint_num = rot_quat.shape[2]

                rot_quat_norm = torch.linalg.norm(rot_quat, dim=3, keepdim=True)
                rot_quat = rot_quat / rot_quat_norm
                rot_mat = quat2mat(rot_quat)
            elif rot_repr == "mat_full":
                rot_mat = motion[..., :-3].reshape((batch_size, -1, 3, 3))
                joint_num = rot_mat.shape[2]
            else:
                raise NotImplementedError
            joint_pos = torch.empty(rot_mat.shape[:-2] + (3,), device=rot_mat.device)

            joint_pos[:, 0] = trans
            for joint_idx, parent_idx in enumerate(topology):
                if parent_idx == -1:
                    assert joint_idx == 0
                    continue
                joint_pos[:, joint_idx] = \
                    torch.matmul(rot_mat[:, parent_idx].clone(),
                                 joint_offset[:, joint_idx].clone().reshape((-1, 3, 1))).squeeze()
                rot_mat[:, joint_idx] = \
                    torch.matmul(rot_mat[:, parent_idx].clone(), rot_mat[:, joint_idx].clone())
                joint_pos[:, joint_idx] += joint_pos[:, parent_idx]
            return joint_pos, rot_mat

        def calc_foot_slide_penetration(joint_pos, threshold=5, ground_height=0):
            anim_len = joint_pos.shape[0]
            joint_height = joint_pos[:, 1]
            joint_pos_xy = joint_pos[:, [0, 2]]

            joint_vel_xy = np.zeros_like(joint_pos_xy)
            joint_vel_xy[1:] = joint_pos_xy[1:] - joint_pos_xy[:-1]
            joint_vel = np.linalg.norm(joint_vel_xy, axis=1)
            up_foot_mask = joint_height < threshold
            penetration_mask = joint_height < ground_height
            penetration = np.mean(penetration_mask * joint_height)
            slide = np.mean(up_foot_mask * joint_vel)
            return slide, penetration

        if self.args.EVAL.calc_foot_slide:
            pred_left_penetration_meter = AverageMeter()
            pred_left_slide_meter = AverageMeter()
            pred_right_penetration_meter = AverageMeter()
            pred_right_slide_meter = AverageMeter()

            gt_left_penetration_meter = AverageMeter()
            gt_left_slide_meter = AverageMeter()
            gt_right_penetration_meter = AverageMeter()
            gt_right_slide_meter = AverageMeter()

        if self.args.EVAL.export_res:
            for item in self.test_result_list:
                name = item["name"]
                export_dir = os.path.join(self.args.LOG.save_dir, self.args.LOG.res_dir)
                gt_out_path = os.path.join(export_dir, "{}_{}_GT.bvh".format(name[0][0], name[1][0]))
                pred_out_path = os.path.join(export_dir, "{}_{}_PRED.bvh".format(name[0][0], name[1][0]))

                gt_out_npz_path = os.path.join(export_dir, "{}_{}_GT.npz".format(name[0][0], name[1][0]))
                pred_out_npz_path = os.path.join(export_dir, "{}_{}_PRED.npz".format(name[0][0], name[1][0]))

                pred_motion = item["pred_motion"].cpu().numpy()
                gt_motion = item["gt_motion"].cpu().numpy()

                pred_trans = pred_motion[:, -3:]
                gt_trans = gt_motion[:, -3:]

                pred_rot_mat = pred_motion[:, :-3].reshape((pred_motion.shape[0], -1, 3, 3))
                gt_rot_mat = gt_motion[:, :-3].reshape((pred_motion.shape[0], -1, 3, 3))

                global_trans = item["global_trans"].cpu().numpy()
                global_rot_mat = item["global_rot_mat"].cpu().numpy()
                align_transform_mat = construct_transform_matrix(global_rot_mat, global_trans)
                pred_aligned_root_mat = construct_transform_matrix(pred_rot_mat[:, 0], pred_trans)
                gt_aligned_root_mat = construct_transform_matrix(gt_rot_mat[:, 0], gt_trans)

                pred_root_transform_mat = align_transform_mat @ pred_aligned_root_mat
                gt_root_transform_mat = align_transform_mat @ gt_aligned_root_mat

                pred_trans = pred_root_transform_mat[:, :3, 3]
                gt_trans = gt_root_transform_mat[:, :3, 3]

                pred_rot_mat[:, 0] = pred_root_transform_mat[:, :3, :3]
                gt_rot_mat[:, 0] = gt_root_transform_mat[:, :3, :3]

                parent = copy.deepcopy(self.args.DATA.parent_list)
                parent[0] = -1
                parent = np.asarray(parent)
                joint_name_list = list(self.args.DATA.joint_name_list)

                pred_joint_offset = item["pred_joint_offset"].cpu().numpy()
                gt_joint_offset = item["gt_joint_offset"].cpu().numpy()

                pred_joint_offset = np.mean(pred_joint_offset, axis=0)
                gt_joint_offset = np.mean(gt_joint_offset, axis=0)

                pred_t_pose = offset2tpose(pred_joint_offset, parent)
                gt_t_pose = offset2tpose(gt_joint_offset, parent)

                if self.args.EVAL.calc_foot_slide:
                    right_foot_joint_idx = 16
                    left_foot_joint_idx = 17
                    global_pred_motion = np.concatenate([pred_rot_mat.reshape((pred_rot_mat.shape[0], -1)),
                                                         pred_trans], axis=1).astype(np.float32)
                    pred_joint_pos = calc_joint_pos_trans(parent, torch.from_numpy(global_pred_motion),
                                                          torch.from_numpy(pred_joint_offset[np.newaxis, :]))[0].numpy()

                    pred_left_foot_joint_pos = pred_joint_pos[:, left_foot_joint_idx]
                    pred_right_foot_joint_pos = pred_joint_pos[:, right_foot_joint_idx]

                    pred_left_slide_penetration = calc_foot_slide_penetration(pred_left_foot_joint_pos)
                    pred_right_slide_penetration = calc_foot_slide_penetration(pred_right_foot_joint_pos)

                    pred_left_slide_meter.update(pred_left_slide_penetration[0])
                    pred_left_penetration_meter.update(pred_left_slide_penetration[1])
                    pred_right_slide_meter.update(pred_right_slide_penetration[0])
                    pred_right_penetration_meter.update(pred_right_slide_penetration[1])

                    global_gt_motion = np.concatenate([gt_rot_mat.reshape((gt_rot_mat.shape[0], -1)),
                                                       gt_trans], axis=1).astype(np.float32)
                    gt_joint_pos = calc_joint_pos_trans(parent, torch.from_numpy(global_gt_motion),
                                                        torch.from_numpy(gt_joint_offset[np.newaxis, :]))[0].numpy()

                    gt_left_foot_joint_pos = gt_joint_pos[:, left_foot_joint_idx]
                    gt_right_foot_joint_pos = gt_joint_pos[:, right_foot_joint_idx]

                    gt_left_slide_penetration = calc_foot_slide_penetration(gt_left_foot_joint_pos)
                    gt_right_slide_penetration = calc_foot_slide_penetration(gt_right_foot_joint_pos)

                    gt_left_slide_meter.update(gt_left_slide_penetration[0])
                    gt_left_penetration_meter.update(gt_left_slide_penetration[1])
                    gt_right_slide_meter.update(gt_right_slide_penetration[0])
                    gt_right_penetration_meter.update(gt_right_slide_penetration[1])

                export_to_bvh(gt_out_path, joint_name_list, gt_rot_mat, gt_trans, gt_t_pose, parent)
                export_to_bvh(pred_out_path, joint_name_list, pred_rot_mat, pred_trans, pred_t_pose, parent)

                np.savez(gt_out_npz_path, joint_names=joint_name_list,
                         rot_mat_arr=gt_rot_mat, trans=gt_trans,
                         t_pose=gt_t_pose, parent=parent)
                np.savez(pred_out_npz_path, joint_names=joint_name_list,
                         rot_mat_arr=pred_rot_mat, trans=pred_trans,
                         t_pose=pred_t_pose, parent=parent)
                print("{} {} {}".format(item["name"][0][0], item["name"][1][0], item["mean_angle_error"]))

        if self.args.EVAL.calc_foot_slide:
            print("pred_left_penetration {} slide {}".format(pred_left_penetration_meter.avg,
                                                             pred_left_slide_meter.avg))
            print("pred_right_penetration {} slide {}".format(pred_right_penetration_meter.avg,
                                                              pred_right_slide_meter.avg))

            print("gt_left_penetration {} slide {}".format(gt_left_penetration_meter.avg,
                                                           gt_left_slide_meter.avg))
            print("gt_right_penetration {} slide {}".format(gt_right_penetration_meter.avg,
                                                            gt_right_slide_meter.avg))
        self.log_loss("test")
        self.test_result_list = []

    def model_forward(self, batch):
        if self.args.DATA.rot_repr == "mat_full":
            (marker_pos, marker_vis, gt_joint_offset,
             gt_motion, gt_joint_pos,
             name, global_rot_mat, global_trans) = \
                (batch[0], batch[1], batch[2],
                 batch[3], batch[4], batch[5],
                 batch[6], batch[7])
        else:
            if self.use_wrist:
                (marker_pos, marker_vis, wrist_marker_pos,
                 gt_joint_offset, gt_motion, gt_joint_pos,
                 gt_phi, name, global_rot_mat, global_trans) = \
                    (batch[0], batch[1], batch[2],
                     batch[3], batch[4], batch[5],
                     batch[6], batch[7], batch[8], batch[9])
            else:
                (marker_pos, marker_vis, gt_joint_offset,
                 gt_motion, gt_joint_pos, gt_phi,
                 name, global_rot_mat, global_trans) = \
                    (batch[0], batch[1], batch[2],
                     batch[3], batch[4], batch[5],
                     batch[6], batch[7], batch[8])

        if self.args.EVAL.export_res:
            marker_pos = cat_window_arr(marker_pos[0])
            marker_vis = cat_window_arr(marker_vis[0])
            gt_joint_offset = gt_joint_offset[0]
            gt_joint_offset = gt_joint_offset[:1].repeat([marker_pos.shape[0], 1, 1])
            gt_motion = cat_window_arr(gt_motion[0])
            gt_joint_pos = cat_window_arr(gt_joint_pos[0])
            if self.args.DATA.rot_repr == "joint_pos":
                gt_phi = cat_window_arr(gt_phi[0])
            global_rot_mat = cat_window_arr(global_rot_mat[0])
            global_trans = cat_window_arr(global_trans[0])

        batch_size, window_shape, marker_num = \
            marker_pos.shape[0], marker_pos.shape[1], marker_pos.shape[2]

        if self.use_window and not self.args.EVAL.export_res:
            batch_size, window_shape, marker_num = \
                marker_pos.shape[0], marker_pos.shape[1], marker_pos.shape[2]
            marker_pos = marker_pos.reshape((-1, *marker_pos.shape[2:]))
            marker_vis = marker_vis.reshape((-1, *marker_vis.shape[2:]))
            gt_joint_offset = gt_joint_offset.unsqueeze(1).repeat([1, window_shape, 1, 1]). \
                view((-1, *gt_joint_offset.shape[1:]))
            gt_motion = gt_motion.reshape((-1, *gt_motion.shape[2:]))
            gt_joint_pos = gt_joint_pos.reshape((-1, *gt_joint_pos.shape[2:]))
            if self.args.DATA.rot_repr == "joint_pos":
                gt_phi = gt_phi.reshape((-1, *gt_phi.shape[2:]))

        if self.use_wrist:
            raise NotImplementedError
            # pred = self.network(
            #     torch.cat([torch.cat([wrist_marker_pos,
            #                           torch.zeros((batch_size, window_shape, 4, 1), device=marker_pos.device)], dim=-1),
            #                torch.cat([marker_pos,
            #                           torch.unsqueeze(marker_vis, dim=-1).repeat((1, 1, 1, 1))], dim=-1)], dim=2). \
            #         view((batch_size, window_shape, -1)))
        else:
            network_input = torch.cat([marker_pos,
                                       torch.unsqueeze(marker_vis, dim=-1)], dim=-1)
            pred = self.network(network_input)

        marker_occ = torch.sum(marker_vis, dim=1)
        marker_occ[marker_occ > 0] = 1

        # pred = pred.view((batch_size, window_shape, -1))

        if self.args.DATA.rot_repr == "mat_full":
            pred_motion = torch.cat(
                [pred[:, :-3].view((batch_size * window_shape, self.args.DATA.joint_num, -1))[:, :, :9].
                 reshape((batch_size * window_shape, -1)), pred[:, -3:]], dim=-1)
            pred_joint_offset = pred[:, :-3].view((batch_size * window_shape,
                                                   self.args.DATA.joint_num, -1))[:, :, -3:]. \
                reshape((batch_size * window_shape, -1, 3))

            motion_std = self.motion_std.to(batch[0].device)
            motion_mean = self.motion_mean.to(batch[0].device)
            joint_offset_mean = self.joint_offset_mean.to(batch[0].device)
            joint_offset_std = self.joint_offset_std.to(batch[0].device)

            loss_dict = self.calc_loss(self.args, pred_motion * motion_std + motion_mean,
                                       gt_motion, pred_joint_offset * joint_offset_std + joint_offset_mean,
                                       gt_joint_offset, gt_joint_pos, marker_occ)
        elif self.args.DATA.rot_repr == "joint_pos":
            pred = pred.view((marker_pos.shape[0], self.args.DATA.joint_num, -1))
            pred_joint_pos = pred[..., :3]
            pred_joint_offset = pred[..., 3:6]
            pred_phi = pred[..., 6:]

            joint_pos_mean = self.joint_pos_mean.to(batch[0].device)
            joint_pos_std = self.joint_pos_std.to(batch[0].device)
            joint_offset_mean = self.joint_offset_mean.to(batch[0].device)
            joint_offset_std = self.joint_offset_std.to(batch[0].device)

            phi_mean = self.phi_mean.to(batch[0].device)
            phi_std = self.phi_std.to(batch[0].device)

            loss_dict = self.calc_loss_ik(self.args, pred_joint_pos * joint_pos_std + joint_pos_mean,
                                          gt_motion, pred_joint_offset * joint_offset_std + joint_offset_mean,
                                          gt_joint_offset, pred_phi * phi_std + phi_mean, gt_phi, gt_joint_pos,
                                          marker_occ)
            loss_dict["name"] = name
            loss_dict["global_rot_mat"] = global_rot_mat
            loss_dict["global_trans"] = global_trans

            # print("WARNING, USING GROUND TRUTH, NO GRAD")
            # loss_dict = self.calc_loss_ik(self.args, gt_joint_pos,
            #                               gt_motion, gt_joint_offset,
            #                               gt_joint_offset, gt_phi, gt_phi, gt_joint_pos,
            #                               marker_occ)

        else:
            raise NotImplementedError

        return loss_dict

    def calc_loss(self, args: argparse.Namespace, pred_motion: torch.Tensor, gt_motion: torch.Tensor,
                  pred_joint_offset: torch.Tensor, gt_joint_offset: torch.Tensor, pre_cal_gt_joint_pos: torch.Tensor,
                  marker_vis: torch.Tensor,
                  trans_lambda: float = 0.1):
        def calc_joint_pos_trans(topology, motion, joint_offset, rot_repr="mat_full"):
            batch_size = motion.shape[0]
            trans = motion[..., -3:]
            if rot_repr == "quat":
                rot_quat = motion[..., :-3].reshape((batch_size, -1, 4))
                joint_num = rot_quat.shape[2]

                rot_quat_norm = torch.linalg.norm(rot_quat, dim=3, keepdim=True)
                rot_quat = rot_quat / rot_quat_norm
                rot_mat = quat2mat(rot_quat)
            elif rot_repr == "mat_full":
                rot_mat = motion[..., :-3].reshape((batch_size, -1, 3, 3))
                joint_num = rot_mat.shape[2]
            else:
                raise NotImplementedError
            joint_pos = torch.empty(rot_mat.shape[:-2] + (3,), device=rot_mat.device)

            joint_pos[:, 0] = trans
            for joint_idx, parent_idx in enumerate(topology):
                if parent_idx == -1:
                    assert joint_idx == 0
                    continue
                joint_pos[:, joint_idx] = \
                    torch.matmul(rot_mat[:, parent_idx].clone(),
                                 joint_offset[:, joint_idx].clone().reshape((-1, 3, 1))).squeeze()
                rot_mat[:, joint_idx] = \
                    torch.matmul(rot_mat[:, parent_idx].clone(), rot_mat[:, joint_idx].clone())
                joint_pos[:, joint_idx] += joint_pos[:, parent_idx]
            return joint_pos, rot_mat

        batch_size = pred_motion.shape[0]
        occ_position = torch.where(marker_vis)

        # pred_mean_joint_offset = torch.mean(pred_joint_offset, dim=1)

        joint_offset_mask = torch.zeros_like(gt_joint_offset)
        joint_offset_mask[torch.abs(gt_joint_offset - 0) > 1e-6] = 1
        # pred_mean_joint_offset = pred_mean_joint_offset * joint_offset_mask

        if args.DATA.rot_repr == "quat":
            pred_rot = pred_motion[:, :, :-3].view((pred_motion.shape[0], pred_motion.shape[1], -1, 4))
            gt_rot = gt_motion[:, :, :-3].view((gt_motion.shape[0], gt_motion.shape[1], -1, 4))

            pred_trans = pred_motion[:, :, -3:]
            gt_trans = gt_motion[:, :, -3:]

            motion_mean_squared_error = torch.mean(
                torch.cat([torch.linalg.norm(pred_rot - gt_rot, dim=3),
                           trans_lambda * torch.linalg.norm(pred_trans - gt_trans, dim=2, keepdim=True)],
                          dim=2))
            normed_pred_rot = pred_rot / torch.unsqueeze(torch.linalg.norm(pred_rot, dim=3), dim=3)
            angle_error = torch.arccos(torch.abs(torch.sum((normed_pred_rot * gt_rot), dim=3))) * 2 * (180 / np.pi)
            mean_angle_error = nanmean(angle_error)
            trans_error = torch.mean(torch.linalg.norm(pred_trans - gt_trans, dim=2))

            occ_mean_squared_error = torch.mean(
                torch.cat([torch.linalg.norm(pred_rot - gt_rot, dim=3),
                           trans_lambda * torch.linalg.norm(pred_trans - gt_trans, dim=2, keepdim=True)],
                          dim=2)[occ_position])
            occ_mean_angle_error = nanmean(angle_error[occ_position])
            occ_trans_error = torch.mean(torch.linalg.norm(pred_trans - gt_trans, dim=2)[occ_position])
        elif args.DATA.rot_repr == "mat_full":
            pred_rot = pred_motion[:, :-3].view((pred_motion.shape[0], -1, 3, 3))
            # gt_rot_quat = gt_motion[:, :, :-3].view((gt_motion.shape[0], gt_motion.shape[1], -1, 4))
            # gt_rot_mat = quat2mat(gt_rot_quat)

            gt_rot_mat = gt_motion[:, :-3].view((gt_motion.shape[0], -1, 3, 3))

            # orthogonalization
            ortho_pred_rot = pred_rot.clone().detach()
            ortho_pred_rot[:, :, :, 1] -= \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[:, :, :, 0], ortho_pred_rot[:, :, :, 1]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[:, :, :, 0], ortho_pred_rot[:, :, :, 0])). \
                    view((batch_size, -1, 1)) * ortho_pred_rot[:, :, :, 0]
            ortho_pred_rot[:, :, :, 2] -= \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[:, :, :, 0], ortho_pred_rot[:, :, :, 2]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[:, :, :, 0], ortho_pred_rot[:, :, :, 0])). \
                    view((batch_size, -1, 1)) * ortho_pred_rot[:, :, :, 0] + \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[:, :, :, 1], ortho_pred_rot[:, :, :, 2]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[:, :, :, 1], ortho_pred_rot[:, :, :, 1])). \
                    view((batch_size, -1, 1)) * ortho_pred_rot[:, :, :, 1]
            ortho_pred_rot[:, :, :, 0] /= torch.linalg.norm(ortho_pred_rot[:, :, :, 0], dim=-1, keepdim=True)
            ortho_pred_rot[:, :, :, 1] /= torch.linalg.norm(ortho_pred_rot[:, :, :, 1], dim=-1, keepdim=True)
            ortho_pred_rot[:, :, :, 2] /= torch.linalg.norm(ortho_pred_rot[:, :, :, 2], dim=-1, keepdim=True)

            pred_trans = pred_motion[:, -3:]
            gt_trans = gt_motion[:, -3:]

            motion_mean_squared_error = torch.mean(
                torch.cat([torch.linalg.norm(pred_rot - gt_rot_mat, dim=[2, 3]),
                           trans_lambda * torch.linalg.norm(pred_trans - gt_trans, dim=1, keepdim=True)],
                          dim=1))
            angle_error = rotmat2angle((torch.linalg.inv(ortho_pred_rot) @ gt_rot_mat).view((-1, 3, 3))). \
                view((batch_size, -1))
            mean_angle_error = nanmean(angle_error)
            trans_error = torch.mean(torch.linalg.norm(pred_trans - gt_trans, dim=-1))

            parent_list = copy.deepcopy(args.DATA.parent_list)
            parent_list[0] = -1

            gt_joint_pos, gt_joint_trans_mat = \
                calc_joint_pos_trans(parent_list, gt_motion, gt_joint_offset, rot_repr="mat_full")
            pred_motion_mat_trans = \
                torch.cat([ortho_pred_rot.view((batch_size, -1)), pred_trans], dim=-1)

            pred_fk_joint_pos, pred_fk_joint_trans_mat = \
                calc_joint_pos_trans(parent_list, pred_motion_mat_trans,
                                     pred_joint_offset, rot_repr="mat_full")

            fk_joint_pos_error = torch.mean(torch.linalg.norm(pred_fk_joint_pos - gt_joint_pos, dim=-1))

            pred_joint_offset_length = torch.linalg.norm(pred_joint_offset, axis=-1)
            gt_joint_offset_length = torch.linalg.norm(gt_joint_offset, axis=-1)

            joint_offset_error = torch.mean(torch.abs(pred_joint_offset_length - gt_joint_offset_length))

            occ_mean_squared_error = torch.mean(
                torch.cat([torch.linalg.norm(pred_rot - gt_rot_mat, dim=[2, 3]),
                           trans_lambda * torch.linalg.norm(pred_trans - gt_trans, dim=1, keepdim=True)],
                          dim=1)[occ_position])
            occ_mean_angle_error = nanmean(angle_error[occ_position])
            occ_trans_error = torch.mean(torch.linalg.norm(pred_trans - gt_trans, dim=1)[occ_position])

            total_loss = motion_mean_squared_error * args.TRAIN.motion_loss_weight + \
                         joint_offset_error * args.TRAIN.joint_offset_loss_weight + \
                         fk_joint_pos_error * args.TRAIN.joint_pos_loss_weight

        else:
            raise NotImplementedError

        loss_dict = {
            "total_loss": total_loss,
            "motion_mean_squared_error": motion_mean_squared_error,
            "mean_angle_error": mean_angle_error,
            "trans_error": trans_error,
            "occ_mean_squared_error": occ_mean_squared_error,
            "occ_mean_angle_error": occ_mean_angle_error,
            "occ_trans_error": occ_trans_error,
            "fk_joint_pos_error": fk_joint_pos_error,
            "joint_offset_error": joint_offset_error}
        return loss_dict

    def calc_loss_ik(self, args: argparse.Namespace, pred_motion: torch.Tensor, gt_motion: torch.Tensor,
                     pred_joint_offset: torch.Tensor, gt_joint_offset: torch.Tensor, pred_phi, gt_phi,
                     pre_cal_gt_joint_pos: torch.Tensor,
                     marker_vis: torch.Tensor,
                     trans_lambda: float = 0.1):
        def calc_joint_pos_trans(topology, motion, joint_offset, rot_repr="mat_full"):
            batch_size = motion.shape[0]
            trans = motion[..., -3:]
            if rot_repr == "quat":
                rot_quat = motion[..., :-3].reshape((batch_size, -1, 4))
                joint_num = rot_quat.shape[2]

                rot_quat_norm = torch.linalg.norm(rot_quat, dim=3, keepdim=True)
                rot_quat = rot_quat / rot_quat_norm
                rot_mat = quat2mat(rot_quat)
            elif rot_repr == "mat_full":
                rot_mat = motion[..., :-3].reshape((batch_size, -1, 3, 3))
                joint_num = rot_mat.shape[2]
            else:
                raise NotImplementedError
            joint_pos = torch.empty(rot_mat.shape[:-2] + (3,), device=rot_mat.device)

            joint_pos[:, 0] = trans
            for joint_idx, parent_idx in enumerate(topology):
                if parent_idx == -1:
                    assert joint_idx == 0
                    continue
                joint_pos[:, joint_idx] = \
                    torch.matmul(rot_mat[:, parent_idx].clone(),
                                 joint_offset[:, joint_idx].clone().reshape((-1, 3, 1))).squeeze()
                rot_mat[:, joint_idx] = \
                    torch.matmul(rot_mat[:, parent_idx].clone(), rot_mat[:, joint_idx].clone())
                joint_pos[:, joint_idx] += joint_pos[:, parent_idx]
            return joint_pos, rot_mat

        assert args.DATA.rot_repr == "joint_pos"

        batch_size = pred_motion.shape[0]
        occ_position = torch.where(marker_vis)

        joint_offset_mask = torch.zeros_like(gt_joint_offset)
        joint_offset_mask[torch.abs(gt_joint_offset - 0) > 1e-6] = 1
        pred_joint_offset = pred_joint_offset * joint_offset_mask

        # pred_mean_joint_offset = pred_mean_joint_offset * joint_offset_mask

        pred_phi = pred_phi / (torch.norm(pred_phi, dim=-1, keepdim=True) + 1e-8)

        phi_mask = torch.ones_like(pred_phi)
        phi_mask[:, args.DATA.no_twist_joint_idx] = 0

        pred_phi[:, np.asarray(args.DATA.no_twist_joint_idx), 0] = 1
        pred_phi[:, np.asarray(args.DATA.no_twist_joint_idx), 1] = 0

        pred_joint_pos = pred_motion.view((batch_size, -1, 3))

        gt_rot_mat = gt_motion[:, :-3].view((gt_motion.shape[0], -1, 3, 3)).clone()
        gt_trans = gt_motion[:, -3:].clone()

        parent_list = copy.deepcopy(args.DATA.parent_list)
        parent_list[0] = -1

        gt_joint_pos, gt_joint_trans_mat = \
            calc_joint_pos_trans(parent_list.copy(), gt_motion.clone(), gt_joint_offset.clone(),
                                 rot_repr="mat_full")

        parent_list = np.asarray(parent_list)

        # pred_rot_mat, calc_joint_pos = \
        #     inverse_kinematics_naive(args, pred_joint_pos, pred_phi, pred_joint_offset,
        #                              parent_list.copy(), svd=False, detach=False)

        pred_rot_mat, calc_joint_pos = \
            inverse_kinematics(args, pred_joint_pos, pred_phi, pred_joint_offset,
                               parent_list.copy(), svd=False, detach=False)

        pred_motion = copy.deepcopy(gt_motion)
        pred_motion[:, :-3] = pred_rot_mat.view(gt_motion[:, :-3].shape)

        fk_joint_pos, fk_joint_trans_mat = \
            calc_joint_pos_trans(parent_list.copy(), pred_motion.clone(), pred_joint_offset.clone(),
                                 rot_repr="mat_full")

        angle_error = rotmat2angle(torch.matmul(pred_rot_mat.swapaxes(2, 3), gt_rot_mat).view((-1, 3, 3)), clip=True). \
            view((batch_size, -1))
        angle_error_mask = torch.ones_like(angle_error)
        angle_error_mask[:, 25] = 0
        angle_error_mask[:, 26] = 0
        angle_error = angle_error * angle_error_mask
        mean_angle_error = torch.nanmean(angle_error)

        angle_loss = torch.sum(torch.sum((pred_rot_mat - gt_rot_mat) ** 2, dim=-1), dim=-1)
        angle_loss[:, 25] *= 0
        angle_loss[:, 26] *= 0
        angle_loss = torch.mean(angle_loss)

        estimate_joint_pos_error = torch.mean(torch.linalg.norm(pred_joint_pos - gt_joint_pos, dim=-1))
        fk_joint_pos_error = torch.mean(torch.linalg.norm(fk_joint_pos - gt_joint_pos, dim=-1))

        pred_joint_offset_length = torch.linalg.norm(pred_joint_offset, axis=-1)

        gt_joint_offset_length = torch.linalg.norm(gt_joint_offset, axis=-1)

        joint_offset_error = torch.mean(torch.abs(pred_joint_offset_length - gt_joint_offset_length))

        phi_error = torch.mean(((pred_phi - gt_phi) * phi_mask) ** 2)

        motion_mean_squared_error = torch.zeros([])
        trans_error = torch.zeros([])
        occ_mean_squared_error = torch.zeros([])
        occ_mean_angle_error = torch.zeros([])
        occ_trans_error = torch.zeros([])

        total_loss = estimate_joint_pos_error * args.TRAIN.joint_pos_loss_weight + \
                     joint_offset_error * args.TRAIN.joint_offset_loss_weight + \
                     phi_error * args.TRAIN.phi_loss_weight + \
                     angle_loss * args.TRAIN.angle_error_weight

        loss_dict = {
            "total_loss": total_loss,
            "motion_mean_squared_error": motion_mean_squared_error,
            "mean_angle_error": mean_angle_error,
            "trans_error": trans_error,
            "occ_mean_squared_error": occ_mean_squared_error,
            "occ_mean_angle_error": occ_mean_angle_error,
            "occ_trans_error": occ_trans_error,
            "estimate_joint_pos_error": estimate_joint_pos_error,
            "fk_joint_pos_error": fk_joint_pos_error,
            "joint_offset_error": joint_offset_error,
            "phi_error": phi_error,

            "gt_motion": gt_motion,
            "pred_motion": pred_motion,
            "gt_joint_offset": gt_joint_offset,
            "pred_joint_offset": pred_joint_offset
        }
        return loss_dict
