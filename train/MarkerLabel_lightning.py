import time

from network.networks import *
from utils.utils import *

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import network
import dataloader
import lightning_module

from utils.parser_util import parse_args_from_yaml


def main():
    # torch.autograd.set_detect_anomaly(True)

    args = parse_args_from_yaml()
    pl.seed_everything(args.SEED)
    log_dir, checkpoint_dir, code_dir, res_dir = create_log(args)

    loggers = [
        pl_loggers.TensorBoardLogger(
            save_dir=log_dir,
            sub_dir='tensorboard',
            version='',
            name=args.NAME
        )
    ]

    progress_bar_callback = pl.callbacks.RichProgressBar()
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_total_loss:.2f}-{val_accuracy:.4f}-{val_f1:.4f}',
        save_top_k=-1,
        every_n_epochs=args.TRAIN.save_interval,
        save_last=False,
    )
    best_model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_total_loss:.2f}-{val_accuracy:.4f}-{val_f1:.4f}',
        monitor="val_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        progress_bar_callback,
        model_checkpoint_callback,
        best_model_checkpoint_callback
    ]
    if "early_stopping" in args.TRAIN.keys():
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=args.TRAIN.early_stopping.monitor,
            min_delta=args.TRAIN.early_stopping.min_delta,
            patience=args.TRAIN.early_stopping.patience,
            verbose=args.TRAIN.early_stopping.verbose,
            mode=args.TRAIN.early_stopping.mode
        )
        callbacks.append(early_stop_callback)

    if len(args.DEVICE) > 1:
        ddp_strategy = 'ddp'
    else:
        ddp_strategy = 'auto'

    trainer = pl.Trainer(
        max_epochs=args.TRAIN.epoch_num,
        accelerator='gpu',
        devices=args.DEVICE,
        strategy=ddp_strategy,
        default_root_dir=args.LOG.save_dir,
        log_every_n_steps=args.TRAIN.log_interval,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        num_sanity_val_steps=2,
        # fast_dev_run=True,
        # detect_anomaly=True
    )

    net = network.build_network(args)

    data_module = dataloader.build_dataloaders(args)

    module = lightning_module.build_lightning_module(args, net, data_module)

    if args.EVAL.only_eval:
        trainer.test(model=module, datamodule=data_module, ckpt_path=os.path.join(args.DATA.root_dir,
                                                                                  args.TRAIN.resume_checkpoint))
        return

    pl.seed_everything(args.SEED, workers=True)
    trainer.fit(model=module,
                datamodule=data_module)


if __name__ == "__main__":
    main()
