import copy

import numpy as np
import torch
import os
import argparse

from pytorch_lightning import LightningModule
from utils.utils import AverageMeter

from sklearn.metrics import classification_report, confusion_matrix

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import Adam, RMSprop
from tqdm import tqdm

from utils.utils import *
from utils.tracklet import *
from network.ik import *


class MarkerLabelLightningModule(LightningModule):
    def __init__(self, args, network, data_module):
        super().__init__()
        self.args = args
        self.network = network
        self.use_window = args.DATA.use_window

        self.log_interval = args.TRAIN.log_interval
        self.save_interval = args.TRAIN.save_interval
        self.resume_checkpoint = args.TRAIN.resume_checkpoint

        self.validation_result_list = []
        self.test_result_list = []

    def configure_optimizers(self):
        optimizer_args = self.args.TRAIN.optimizer
        lr_scheduler_args = self.args.TRAIN.lr_scheduler

        optimize_params = [a[1] for a in self.network.named_parameters() if a[1].requires_grad]
        if optimizer_args.type == "Adam":
            optimizer = Adam(optimize_params, **optimizer_args.args)
        else:
            raise NotImplementedError

        if lr_scheduler_args.type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_args.args)
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
            "total_loss": AverageMeter()}
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
        loss_dict, drec = self.model_forward(batch)
        self.update_loss(loss_dict, "train")
        for item in loss_dict:
            self.log("train_{}".format(item), loss_dict[item].item(),
                     on_step=True, on_epoch=False, prog_bar=True)

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, drec = self.model_forward(batch)

        self.update_loss(loss_dict, "val")
        result = {"label_ids_gt": copy2cpu(batch["label_ids"]),
                  "label_ids_pred": copy2cpu(drec["label_ids"])}

        if self.args.DATA.train_part == "full_body":
            result.update({"part_ids_gt": batch["part_ids"].detach().cpu().numpy(),
                           "part_ids_pred": drec["part_ids"].detach().cpu().numpy()})
        self.validation_result_list.append(result)

    def test_step(self, batch, batch_idx):
        loss_dict, drec = self.model_forward(batch)
        self.update_loss(loss_dict, "test")

        result = {"attention_weights": copy2cpu(drec["attention_weights"]),
                  "feature_dist": copy2cpu(drec["feature_local_dist"]),
                  "perm_vec": copy2cpu(batch["perm_vec"]),
                  "label_confidence": copy2cpu(drec["label_confidence"]),
                  "marker_pos": copy2cpu(batch["marker_pos"]),
                  "label_ids_gt": copy2cpu(batch["label_ids"]),
                  "label_ids_pred": copy2cpu(drec["label_ids"])}
        if self.args.DATA.train_part == "full_body":
            result.update({"part_ids_gt": batch["part_ids"].detach().cpu().numpy(),
                           "part_ids_pred": drec["part_ids"].detach().cpu().numpy()})
        if "point_features" in drec:
            result["point_features"] = copy2cpu(drec["point_features"])

        if self.args.EVAL.export_vicon:
            result["label_ids_vicon"] = copy2cpu(batch["vicon_ids"])

        self.test_result_list.append(result)

    def on_train_epoch_start(self) -> None:
        self.init_loss_meter("train")

    def on_validation_epoch_start(self) -> None:
        self.init_loss_meter("val")

    def on_test_epoch_start(self) -> None:
        self.init_loss_meter("test")

    def on_train_epoch_end(self) -> None:
        self.log_loss("train")

    def on_validation_epoch_end(self) -> None:
        label_metric_dict = {"f1": AverageMeter(),
                             "precision": AverageMeter(),
                             "recall": AverageMeter(),
                             "accuracy": AverageMeter()}
        if self.args.DATA.train_part == "full_body":
            body_label_metric_dict = {"body_f1": AverageMeter(),
                                      "body_precision": AverageMeter(),
                                      "body_recall": AverageMeter(),
                                      "body_accuracy": AverageMeter()}
            left_hand_label_metric_dict = {"left_hand_f1": AverageMeter(),
                                           "left_hand_precision": AverageMeter(),
                                           "left_hand_recall": AverageMeter(),
                                           "left_hand_accuracy": AverageMeter()}
            right_hand_label_metric_dict = {"right_hand_f1": AverageMeter(),
                                            "right_hand_precision": AverageMeter(),
                                            "right_hand_recall": AverageMeter(),
                                            "right_hand_accuracy": AverageMeter()}
            part_metric_dict = {"part_f1": AverageMeter(),
                                "part_precision": AverageMeter(),
                                "part_recall": AverageMeter(),
                                "part_accuracy": AverageMeter()}

        for idx in range(len(self.validation_result_list)):
            res = self.validation_result_list[idx]
            labeling_report = classification_report(res["label_ids_gt"].flatten(),
                                                    res["label_ids_pred"].flatten(),
                                                    output_dict=True, zero_division=0)
            avg_mode = 'macro'

            f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
            precision = labeling_report[f'{avg_mode} avg']['precision']
            recall = labeling_report[f'{avg_mode} avg']['recall']
            if 'accuracy' in labeling_report:
                accuracy = labeling_report['accuracy']
            else:
                accuracy = np.sum(res["label_ids_gt"].flatten() == res["label_ids_pred"].flatten()) / \
                           len(res["label_ids_pred"].flatten())

            label_metric_dict["f1"].update(f1_score)
            label_metric_dict["precision"].update(precision)
            label_metric_dict["recall"].update(recall)
            label_metric_dict["accuracy"].update(accuracy)
            if self.args.DATA.train_part == "full_body":
                def part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids, part_id, label_id_range):
                    part_mask = (part_ids == part_id)

                    part_label_ids_gt = label_ids_gt[part_mask]
                    part_label_ids_pred = label_ids_pred[part_mask]

                    null_id = label_id_range[1] - label_id_range[0]
                    in_range_mask_gt = (part_label_ids_gt >= label_id_range[0]) & \
                                       (part_label_ids_gt < label_id_range[1])
                    part_label_ids_gt[np.bitwise_not(in_range_mask_gt)] = null_id
                    part_label_ids_gt[in_range_mask_gt] -= label_id_range[0]

                    in_range_mask_pred = (part_label_ids_pred >= label_id_range[0]) & \
                                         (part_label_ids_pred < label_id_range[1])
                    part_label_ids_pred[np.bitwise_not(in_range_mask_pred)] = null_id
                    part_label_ids_pred[in_range_mask_pred] -= label_id_range[0]

                    labeling_report = classification_report(part_label_ids_gt.flatten(),
                                                            part_label_ids_pred.flatten(),
                                                            output_dict=True, zero_division=0)

                    f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
                    precision = labeling_report[f'{avg_mode} avg']['precision']
                    recall = labeling_report[f'{avg_mode} avg']['recall']
                    if 'accuracy' in labeling_report:
                        accuracy = labeling_report['accuracy']
                    else:
                        accuracy = np.sum(part_label_ids_gt.flatten() == part_label_ids_pred.flatten()) / \
                                   len(part_label_ids_pred.flatten())
                    return {"f1_score": f1_score,
                            "precision": precision,
                            "recall": recall,
                            "accuracy": accuracy}

                label_ids_gt = res["label_ids_gt"]
                label_ids_pred = res["label_ids_pred"]
                part_ids_gt = res["part_ids_gt"]
                part_ids_pred = res["part_ids_pred"]

                body_part_classification_report = \
                    part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids_gt, 0,
                                               [0, self.args.DATA.body_marker_num])
                left_hand_part_classification_report = \
                    part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids_gt, 1,
                                               [self.args.DATA.body_marker_num,
                                                self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num])
                right_hand_part_classification_report = \
                    part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids_gt, 2,
                                               [self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num,
                                                self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num * 2])

                part_report = classification_report(part_ids_gt.flatten(), part_ids_pred.flatten(),
                                                    output_dict=True, zero_division=0)

                part_f1_score = part_report[f'{avg_mode} avg']['f1-score']
                part_precision = part_report[f'{avg_mode} avg']['precision']
                part_recall = part_report[f'{avg_mode} avg']['recall']
                if 'accuracy' in labeling_report:
                    part_accuracy = part_report['accuracy']
                else:
                    part_accuracy = np.sum(part_ids_gt.flatten() == part_ids_pred.flatten()) / \
                                    len(part_ids_pred.flatten())

                part_metric_dict["part_f1"].update(part_f1_score)
                part_metric_dict["part_precision"].update(part_precision)
                part_metric_dict["part_recall"].update(part_recall)
                part_metric_dict["part_accuracy"].update(part_accuracy)

                body_label_metric_dict["body_f1"]. \
                    update(body_part_classification_report["f1_score"])
                body_label_metric_dict["body_precision"]. \
                    update(body_part_classification_report["precision"])
                body_label_metric_dict["body_recall"]. \
                    update(body_part_classification_report["recall"])
                body_label_metric_dict["body_accuracy"]. \
                    update(body_part_classification_report["accuracy"])

                left_hand_label_metric_dict["left_hand_f1"]. \
                    update(left_hand_part_classification_report["f1_score"])
                left_hand_label_metric_dict["left_hand_precision"]. \
                    update(left_hand_part_classification_report["precision"])
                left_hand_label_metric_dict["left_hand_recall"]. \
                    update(left_hand_part_classification_report["recall"])
                left_hand_label_metric_dict["left_hand_accuracy"]. \
                    update(left_hand_part_classification_report["accuracy"])

                right_hand_label_metric_dict["right_hand_f1"]. \
                    update(right_hand_part_classification_report["f1_score"])
                right_hand_label_metric_dict["right_hand_precision"]. \
                    update(right_hand_part_classification_report["precision"])
                right_hand_label_metric_dict["right_hand_recall"]. \
                    update(right_hand_part_classification_report["recall"])
                right_hand_label_metric_dict["right_hand_accuracy"]. \
                    update(right_hand_part_classification_report["accuracy"])

        if self.args.DATA.train_part == "full_body":
            self.test_meters.update(body_label_metric_dict)
            self.test_meters.update(left_hand_label_metric_dict)
            self.test_meters.update(right_hand_label_metric_dict)
            self.test_meters.update(part_metric_dict)

        self.test_meters.update(label_metric_dict)
        self.log_loss("val")
        self.validation_result_list = []

    def on_test_epoch_end(self) -> None:
        label_metric_dict = {"f1": AverageMeter(),
                             "precision": AverageMeter(),
                             "recall": AverageMeter(),
                             "accuracy": AverageMeter()}

        if self.args.EVAL.export_attention:
            attention_res_path = os.path.join(self.args.LOG.save_dir, self.args.LOG.res_dir, "attention.npy")
            feature_dist_path = os.path.join(self.args.LOG.save_dir, self.args.LOG.res_dir, "feature_dist.npy")
            attention_list = []
            feature_dist_list = []
            for res_idx in range(len(self.test_result_list)):
                res = self.test_result_list[res_idx]
                attention_list.append(res["attention_weights"])
            np.save(attention_res_path, np.mean(np.concatenate(attention_list, axis=0),axis=0))
            del attention_list

            for res_idx in range(len(self.test_result_list)):
                res = self.test_result_list[res_idx]
                feature_dist_list.append(res["feature_dist"])

            np.save(feature_dist_path, np.mean(np.concatenate(feature_dist_list, axis=0),axis=0))

        if self.args.DATA.use_window:
            tracklet_label_metric_dict = {"tracklet_f1": AverageMeter(),
                                          "tracklet_precision": AverageMeter(),
                                          "tracklet_recall": AverageMeter(),
                                          "tracklet_accuracy": AverageMeter()}

        if self.args.EVAL.export_vicon:
            vicon_label_metric_dict = {"vicon_f1": AverageMeter(),
                                       "vicon_precision": AverageMeter(),
                                       "vicon_recall": AverageMeter(),
                                       "vicon_accuracy": AverageMeter()}

        if self.args.DATA.train_part == "full_body":
            body_label_metric_dict = {"body_f1": AverageMeter(),
                                      "body_precision": AverageMeter(),
                                      "body_recall": AverageMeter(),
                                      "body_accuracy": AverageMeter()}
            left_hand_label_metric_dict = {"left_hand_f1": AverageMeter(),
                                           "left_hand_precision": AverageMeter(),
                                           "left_hand_recall": AverageMeter(),
                                           "left_hand_accuracy": AverageMeter()}
            right_hand_label_metric_dict = {"right_hand_f1": AverageMeter(),
                                            "right_hand_precision": AverageMeter(),
                                            "right_hand_recall": AverageMeter(),
                                            "right_hand_accuracy": AverageMeter()}
            part_metric_dict = {"part_f1": AverageMeter(),
                                "part_precision": AverageMeter(),
                                "part_recall": AverageMeter(),
                                "part_accuracy": AverageMeter()}

        # for res_idx in tqdm(range(len(self.test_result_list))):
        for res_idx in range(len(self.test_result_list)):
            res = self.test_result_list[res_idx]

            labeling_report = classification_report(res["label_ids_gt"].flatten(),
                                                    res["label_ids_pred"].flatten(),
                                                    output_dict=True, zero_division=0)
            avg_mode = 'macro'

            f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
            precision = labeling_report[f'{avg_mode} avg']['precision']
            recall = labeling_report[f'{avg_mode} avg']['recall']
            if 'accuracy' in labeling_report:
                accuracy = labeling_report['accuracy']
            else:
                accuracy = np.sum(res["label_ids_gt"].flatten() == res["label_ids_pred"].flatten()) / \
                           len(res["label_ids_pred"].flatten())

            label_metric_dict["f1"].update(f1_score)
            label_metric_dict["precision"].update(precision)
            label_metric_dict["recall"].update(recall)
            label_metric_dict["accuracy"].update(accuracy)

            if self.args.EVAL.export_res:
                out_path = os.path.join(self.args.LOG.save_dir, self.args.LOG.res_dir, "{}.npz".format(res_idx))
                if not self.args.EVAL.export_vicon:
                    np.savez(out_path, label_ids_gt=res["label_ids_gt"],
                             label_ids_pred=res["label_ids_pred"],
                             perm_vec=res["perm_vec"],
                             marker_pos=res["marker_pos"])
                else:
                    np.savez(out_path, label_ids_gt=res["label_ids_gt"],
                             label_ids_pred=res["label_ids_pred"],
                             label_ids_vicon=res["label_ids_vicon"],
                             perm_vec=res["perm_vec"],
                             marker_pos=res["marker_pos"])

            occ_rate = (sum(sum(res["label_ids_gt"] == res["label_ids_gt"].shape[1])) /
                        (res["label_ids_gt"].shape[0] * res["label_ids_gt"].shape[1]))
            print("idx {} f1 {} precision {} recall {} accuracy {} occ_rate {}".
                  format(res_idx, f1_score, precision, recall, accuracy, occ_rate))

            if self.args.EVAL.export_vicon:
                labeling_report = classification_report(res["label_ids_gt"].flatten(),
                                                        res["label_ids_vicon"].flatten(),
                                                        output_dict=True, zero_division=0)
                avg_mode = 'macro'

                f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
                precision = labeling_report[f'{avg_mode} avg']['precision']
                recall = labeling_report[f'{avg_mode} avg']['recall']
                if 'accuracy' in labeling_report:
                    accuracy = labeling_report['accuracy']
                else:
                    accuracy = np.sum(res["label_ids_gt"].flatten() == res["label_ids_vicon"].flatten()) / \
                               len(res["label_ids_vicon"].flatten())

                vicon_label_metric_dict["vicon_f1"].update(f1_score)
                vicon_label_metric_dict["vicon_precision"].update(precision)
                vicon_label_metric_dict["vicon_recall"].update(recall)
                vicon_label_metric_dict["vicon_accuracy"].update(accuracy)

                # occ_rate = (sum(sum(res["label_ids_gt"] == res["label_ids_gt"].shape[1])) /
                #             (res["label_ids_gt"].shape[0] * res["label_ids_gt"].shape[1]))
                # print("idx {} f1 {} precision {} recall {} accuracy {} occ_rate {}".
                #       format(res_idx, f1_score, precision, recall, accuracy, occ_rate))

            if self.args.EVAL.use_tracklet:
                label_ids_gt = res["label_ids_gt"]
                label_ids_pred = res["label_ids_pred"]
                marker_pos = res["marker_pos"]
                label_confidence = res["label_confidence"]
                point_features = res["point_features"]

                label_ids_pred = label_ids_pred.reshape(label_ids_gt.shape)
                label_confidence = label_confidence.reshape((label_ids_pred.shape[0], label_ids_pred.shape[1],
                                                             label_confidence.shape[1], label_confidence.shape[2]))
                window_num, window_size = label_ids_gt.shape[0], label_ids_gt.shape[1]
                point_features = point_features.reshape((window_num, window_size, *point_features.shape[1:]))
                tracklet_label_ids_pred = np.zeros_like(label_ids_pred)
                for window_idx in range(window_num):
                    window_marker_pos = marker_pos[window_idx]
                    window_label_ids_pred = label_ids_pred[window_idx]
                    window_label_confidence = label_confidence[window_idx]
                    window_point_feature = point_features[window_idx]
                    tracklet_label_ids = tracklet_labeling(self.args.EVAL.tracklet,
                                                           window_label_confidence,
                                                           window_marker_pos,
                                                           window_point_feature)

                    tracklet_label_ids_pred[window_idx] = tracklet_label_ids

                labeling_report = classification_report(label_ids_gt.flatten(),
                                                        tracklet_label_ids_pred.flatten(),
                                                        output_dict=True, zero_division=0)
                avg_mode = 'macro'

                f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
                precision = labeling_report[f'{avg_mode} avg']['precision']
                recall = labeling_report[f'{avg_mode} avg']['recall']
                if 'accuracy' in labeling_report:
                    accuracy = labeling_report['accuracy']
                else:
                    accuracy = np.sum(res["label_ids_gt"].flatten() == res["label_ids_pred"].flatten()) / \
                               len(res["label_ids_pred"].flatten())

                tracklet_label_metric_dict["tracklet_f1"].update(f1_score)
                tracklet_label_metric_dict["tracklet_precision"].update(precision)
                tracklet_label_metric_dict["tracklet_recall"].update(recall)
                tracklet_label_metric_dict["tracklet_accuracy"].update(accuracy)

                occ_rate = (sum(sum(res["label_ids_gt"] == res["label_ids_gt"].shape[1])) /
                            (res["label_ids_gt"].shape[0] * res["label_ids_gt"].shape[1]))
                print("idx {} f1 {} precision {} recall {} accuracy {} occ_rate {}".
                      format(res_idx, f1_score, precision, recall, accuracy, occ_rate))

            if self.args.DATA.train_part == "full_body":
                def part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids, part_id, label_id_range):
                    part_mask = (part_ids == part_id)

                    part_label_ids_gt = label_ids_gt[part_mask]
                    part_label_ids_pred = label_ids_pred[part_mask]

                    null_id = label_id_range[1] - label_id_range[0]
                    in_range_mask_gt = (part_label_ids_gt >= label_id_range[0]) & \
                                       (part_label_ids_gt < label_id_range[1])
                    part_label_ids_gt[np.bitwise_not(in_range_mask_gt)] = null_id
                    part_label_ids_gt[in_range_mask_gt] -= label_id_range[0]

                    in_range_mask_pred = (part_label_ids_pred >= label_id_range[0]) & \
                                         (part_label_ids_pred < label_id_range[1])
                    part_label_ids_pred[np.bitwise_not(in_range_mask_pred)] = null_id
                    part_label_ids_pred[in_range_mask_pred] -= label_id_range[0]

                    labeling_report = classification_report(part_label_ids_gt.flatten(),
                                                            part_label_ids_pred.flatten(),
                                                            output_dict=True, zero_division=0)

                    f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
                    precision = labeling_report[f'{avg_mode} avg']['precision']
                    recall = labeling_report[f'{avg_mode} avg']['recall']
                    if 'accuracy' in labeling_report:
                        accuracy = labeling_report['accuracy']
                    else:
                        accuracy = np.sum(part_label_ids_gt.flatten() == part_label_ids_pred.flatten()) / \
                                   len(part_label_ids_pred.flatten())
                    return {"f1_score": f1_score,
                            "precision": precision,
                            "recall": recall,
                            "accuracy": accuracy}

                label_ids_gt = res["label_ids_gt"]
                label_ids_pred = res["label_ids_pred"]
                part_ids_gt = res["part_ids_gt"]
                part_ids_pred = res["part_ids_pred"]

                body_part_classification_report = \
                    part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids_gt, 0,
                                               [0, self.args.DATA.body_marker_num])
                left_hand_part_classification_report = \
                    part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids_gt, 1,
                                               [self.args.DATA.body_marker_num,
                                                self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num])
                right_hand_part_classification_report = \
                    part_classification_report(label_ids_gt, label_ids_pred,
                                               part_ids_gt, 2,
                                               [self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num,
                                                self.args.DATA.body_marker_num + self.args.DATA.hand_marker_num * 2])

                part_report = classification_report(part_ids_gt.flatten(), part_ids_pred.flatten(),
                                                    output_dict=True, zero_division=0)

                part_f1_score = part_report[f'{avg_mode} avg']['f1-score']
                part_precision = part_report[f'{avg_mode} avg']['precision']
                part_recall = part_report[f'{avg_mode} avg']['recall']
                if 'accuracy' in labeling_report:
                    part_accuracy = part_report['accuracy']
                else:
                    part_accuracy = np.sum(part_ids_gt.flatten() == part_ids_pred.flatten()) / \
                                    len(part_ids_pred.flatten())

                part_metric_dict["part_f1"].update(part_f1_score)
                part_metric_dict["part_precision"].update(part_precision)
                part_metric_dict["part_recall"].update(part_recall)
                part_metric_dict["part_accuracy"].update(part_accuracy)

                body_label_metric_dict["body_f1"]. \
                    update(body_part_classification_report["f1_score"])
                body_label_metric_dict["body_precision"]. \
                    update(body_part_classification_report["precision"])
                body_label_metric_dict["body_recall"]. \
                    update(body_part_classification_report["recall"])
                body_label_metric_dict["body_accuracy"]. \
                    update(body_part_classification_report["accuracy"])

                left_hand_label_metric_dict["left_hand_f1"]. \
                    update(left_hand_part_classification_report["f1_score"])
                left_hand_label_metric_dict["left_hand_precision"]. \
                    update(left_hand_part_classification_report["precision"])
                left_hand_label_metric_dict["left_hand_recall"]. \
                    update(left_hand_part_classification_report["recall"])
                left_hand_label_metric_dict["left_hand_accuracy"]. \
                    update(left_hand_part_classification_report["accuracy"])

                right_hand_label_metric_dict["right_hand_f1"]. \
                    update(right_hand_part_classification_report["f1_score"])
                right_hand_label_metric_dict["right_hand_precision"]. \
                    update(right_hand_part_classification_report["precision"])
                right_hand_label_metric_dict["right_hand_recall"]. \
                    update(right_hand_part_classification_report["recall"])
                right_hand_label_metric_dict["right_hand_accuracy"]. \
                    update(right_hand_part_classification_report["accuracy"])

        if self.args.EVAL.export_vicon:
            self.test_meters.update(vicon_label_metric_dict)
        if self.args.DATA.train_part == "full_body":
            self.test_meters.update(body_label_metric_dict)
            self.test_meters.update(left_hand_label_metric_dict)
            self.test_meters.update(right_hand_label_metric_dict)
            self.test_meters.update(part_metric_dict)
        if self.args.DATA.use_window:
            self.test_meters.update(tracklet_label_metric_dict)

        self.test_meters.update(label_metric_dict)
        self.log_loss("test")
        self.test_result_list = []

    def model_forward(self, batch):
        marker_pos, gt_label_ids, gt_aug_asmat, aug_asmat_weights = \
            batch["marker_pos"], batch["label_ids"], \
                batch["aug_asmat"], batch["aug_asmat_weights"]
        if self.use_window:
            marker_pos = marker_pos.reshape((-1, *marker_pos.shape[2:]))
            gt_label_ids = gt_label_ids.reshape((-1, *gt_label_ids.shape[2:]))
            gt_aug_asmat = gt_aug_asmat.reshape((-1, *gt_aug_asmat.shape[2:]))
            aug_asmat_weights = aug_asmat_weights.reshape((-1, *aug_asmat_weights.shape[2:]))

        drec = self.network(marker_pos)
        pred_aug_asmat = drec["aug_asmat"]

        label_assignment_loss = self.calc_label_loss(pred_aug_asmat,
                                                     gt_aug_asmat,
                                                     aug_asmat_weights)
        if self.args.DATA.train_part == "full_body":
            pred_part_scores = drec["part_scores"]
            gt_part_ids = batch["part_ids"]

            part_loss = self.calc_part_loss(pred_part_scores, gt_part_ids)
            total_loss = part_loss * self.args.TRAIN.part_loss_weight + \
                         label_assignment_loss * self.args.TRAIN.label_loss_weight
        else:
            total_loss = label_assignment_loss

        loss_dict = {"total_loss": total_loss}

        return loss_dict, drec

    def calc_part_loss(self, pred_score, gt_ids):
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred_score.reshape((-1, 3)), gt_ids.flatten())

        return loss

    def calc_label_loss(self, pred_aug_asmat, gt_aug_asmat, aug_asmat_weights):
        nominator = pred_aug_asmat * gt_aug_asmat
        weighted_nominator = nominator * aug_asmat_weights

        nll = weighted_nominator.sum([1, 2], keepdim=True) / gt_aug_asmat.sum([1, 2], keepdim=True)

        return -nll.sum()
