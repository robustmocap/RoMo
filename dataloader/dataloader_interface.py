from .marker2pose_dataset import Marker2PoseDataModule
from .markerlabel_dataset import MarkerLabelDataModule

def build_dataloaders(args):
    if args.TYPE == "marker2pose":
        return Marker2PoseDataModule(args)
    elif args.TYPE == "marker_label":
        return MarkerLabelDataModule(args)
    else:
        raise ValueError("Unrecognized dataloader type {}".format(args.DATA.type))
