import json

from .networks import *
from .label_networks import *

BODY_JOINT_NAME_LIST_PATH = "assets/joint_config_body.txt"
LEFTHAND_JOINT_CONFIG_PATH = "assets/joint_config_left_hand.txt"
RIGHTHAND_JOINT_CONFIG_PATH = "assets/joint_config_right_hand.txt"
MARKER_NAME_CONFIG_PATH = "assets/marker_names.json"
with open(BODY_JOINT_NAME_LIST_PATH, "r") as f:
    content = f.readlines()
    BODY_JOINT_NAME_LIST = [line.split()[0] for line in content]
    BODY_JOINT_PARENT = [int(line.split()[1]) for line in content]
    BODY_END_JOINT_IDX = [16, 17, 24, 25, 26]

with open(LEFTHAND_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    LEFTHAND_JOINT_NAME_LIST = [line.split()[0] for line in content]
    LEFTHAND_JOINT_PARENT = [int(line.split()[1]) for line in content]
    LEFTHAND_END_JOINT_IDX = [14, 19, 21, 22, 23]

with open(RIGHTHAND_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    RIGHTHAND_JOINT_NAME_LIST = [line.split()[0] for line in content]
    RIGHTHAND_JOINT_PARENT = [int(line.split()[1]) for line in content]
    RIGHTHAND_END_JOINT_IDX = [14, 19, 21, 22, 23]

marker_name_config = json.load(open(MARKER_NAME_CONFIG_PATH, "r"))
PRODUCTION_MARKER_NAME_LIST = marker_name_config["body"]["production"]
FRONT_WAIST_MARKER_NAME_LIST = marker_name_config["body"]["front_waist"]
THREE_FINGER_MARKER_NAME_LIST = marker_name_config["finger"]["3"]
FIVE_FINGER_MARKER_NAME_LIST = marker_name_config["finger"]["5"]
TEN_FINGER_MARKER_NAME_LIST = marker_name_config["finger"]["10"]


def build_network(args):
    marker_name_config_path = os.path.join(args.DATA.root_dir,
                                           args.DATA.marker_name_config_path)
    marker_name_config = json.load(open(marker_name_config_path, "r"))

    left_hand_joint_name_config_path = os.path.join(args.DATA.root_dir,
                                                    args.DATA.left_hand_joint_name_config_path)
    right_hand_joint_name_config_path = os.path.join(args.DATA.root_dir,
                                                     args.DATA.right_hand_joint_name_config_path)
    body_joint_name_config_path = os.path.join(args.DATA.root_dir,
                                               args.DATA.body_joint_name_config_path)
    full_body_joint_name_config_path = os.path.join(args.DATA.root_dir,
                                                    args.DATA.full_body_joint_name_config_path)

    train_part = args.DATA.train_part
    if train_part in ["left_hand", "right_hand", "hand"]:
        marker_list = marker_name_config["finger"][args.DATA.finger_type]
        marker_num = len(marker_list) // 2
        if train_part == "left_hand":
            with open(left_hand_joint_name_config_path, "r") as f:
                content = f.readlines()
                joint_list = [line.split()[0] for line in content]
                parent_list = LEFTHAND_JOINT_PARENT
                joint_name_list = LEFTHAND_JOINT_NAME_LIST
        elif train_part == "right_hand":
            with open(right_hand_joint_name_config_path, "r") as f:
                content = f.readlines()
                joint_list = [line.split()[0] for line in content]
                parent_list = RIGHTHAND_JOINT_PARENT
                joint_name_list = RIGHTHAND_JOINT_NAME_LIST
        joint_num = len(joint_list)
    elif train_part == "body":
        marker_list = marker_name_config["body"][args.DATA.body_type]
        marker_num = len(marker_list)
        with open(body_joint_name_config_path, "r") as f:
            content = f.readlines()
            joint_list = [line.split()[0] for line in content]
        joint_num = len(joint_list)

        if args.DATA.body_type == "grab":
            parent_list = list(map(int, [line.split()[1] for line in content]))
            joint_name_list = joint_list
        else:
            parent_list = BODY_JOINT_PARENT
            joint_name_list = BODY_JOINT_NAME_LIST
    elif train_part == "full_body":
        marker_list = marker_name_config["body"][args.DATA.body_type] + \
                      marker_name_config["finger"][args.DATA.finger_type]
        marker_num = len(marker_list)
        with open(full_body_joint_name_config_path, "r") as f:
            content = f.readlines()
            joint_list = [line.split()[0] for line in content]
        joint_num = len(joint_list)
        parent_list = BODY_JOINT_PARENT
        joint_name_list = BODY_JOINT_NAME_LIST
    else:
        raise Exception("Wrong training part")

    args.DATA.marker_num = marker_num
    args.DATA.joint_num = joint_num
    args.DATA.parent_list = parent_list
    args.DATA.joint_name_list = joint_name_list

    if args.NETWORK.type == "BiLSTM":
        # used for Marker2Marker
        if args.DATA.use_wrist:
            return BiLSTM(args, (marker_num + 4) * 6,
                          args.NETWORK.hidden_dim, marker_num * 3,
                          act="Mish", layer_num=args.NETWORK.lstm_layer_num)
        else:
            return BiLSTM(args, marker_num * 6,
                          args.NETWORK.hidden_dim, marker_num * 3,
                          act="Mish", layer_num=args.NETWORK.lstm_layer_num)
    elif args.NETWORK.type == "GNNBiLSTM":
        # used for Marker2Marker
        if args.DATA.use_wrist:
            return GNNBiLSTM(args, (marker_num + 4) * 6,
                             args.NETWORK.hidden_dim, marker_num * 3,
                             layer_num=args.NETWORK.lstm_layer_num)
        else:
            return GNNBiLSTM(args, marker_num * 6,
                             args.NETWORK.hidden_dim, marker_num * 3,
                             layer_num=args.NETWORK.lstm_layer_num)
    elif args.NETWORK.type == "HeteroGraphModel":
        # used for Marker2Pose
        if args.DATA.rot_repr in ["quat", "mat_full", "mat_part", "angle"]:
            joint_num_extended = joint_num + 1
        elif args.DATA.rot_repr == "joint_pos":
            joint_num_extended = joint_num
        else:
            raise NotImplementedError

        if args.DATA.use_wrist:
            marker_num_extended = marker_num + 4
        else:
            marker_num_extended = marker_num

        args.NETWORK.network_dim[0] = marker_num_extended * args.NETWORK.marker_gcn_dim[-1]
        args.NETWORK.network_dim[-1] = joint_num_extended * args.NETWORK.skeleton_gcn_dim[0]

        if args.DATA.use_wrist:
            args.DATA.marker_num = marker_num + 4

        if args.DATA.rot_repr == "quat":
            args.NETWORK.skeleton_gcn_dim[-1] = 4
        elif args.DATA.rot_repr == "mat_full":
            args.NETWORK.skeleton_gcn_dim[-1] = 9 + 3
        elif args.DATA.rot_repr == "mat_part":
            args.NETWORK.skeleton_gcn_dim[-1] = 6
        elif args.DATA.rot_repr == "angle":
            args.NETWORK.skeleton_gcn_dim[-1] = 3
        elif args.DATA.rot_repr == "joint_pos":
            # joint_pos, joint_offset, phi
            args.NETWORK.skeleton_gcn_dim[-1] = (3 + 3 + 2)
        else:
            raise NotImplementedError
        return HeteroGraphModel(args)
    elif args.NETWORK.type == "SOMA":
        return SOMA_Network(args)
    elif args.NETWORK.type == "PCT":
        return PCT_Labeller(args)
    else:
        raise NotImplementedError
