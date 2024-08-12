from .marker2pose_lightning_module import Marker2PoseLightningModule
from .marker_label_lightning_module import MarkerLabelLightningModule

import os


def build_lightning_module(args, network, data_module):
    if args.TYPE == "marker2pose":
        module = Marker2PoseLightningModule
    elif args.TYPE == "marker_label":
        module = MarkerLabelLightningModule
    else:
        raise ValueError("Unrecognized dataloader type {}".format(args.DATA.type))

    if len(args.TRAIN.resume_checkpoint):
        resume_checkpoint_path = os.path.join(args.DATA.root_dir,
                                              args.TRAIN.resume_checkpoint)
        print("Loading checkpoint {}".format(resume_checkpoint_path))
        return module.load_from_checkpoint(resume_checkpoint_path,
                                           args=args, network=network, data_module=data_module)
    else:
        return module(args, network, data_module)
