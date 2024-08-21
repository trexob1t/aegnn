import argparse
import datetime
import os
import pytorch_lightning as pl
import pytorch_lightning.loggers
import wandb

import aegnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model name to train.")
    parser.add_argument("--task", type=str, required=True, help="Task to perform, such as detection or recognition.")
    parser.add_argument("--dim", type=int, help="Dimensionality of input data", default=3)
    parser.add_argument("--seed", default=12345, type=int)

    # Add this line to include a checkpoint argument
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific checkpoint to load")

    group = parser.add_argument_group("Trainer")
    group.add_argument("--max-epochs", default=150, type=int)
    group.add_argument("--overfit-batches", default=0.0, type=int)
    group.add_argument("--log-every-n-steps", default=10, type=int)
    group.add_argument("--gradient_clip_val", default=0.0, type=float)
    group.add_argument("--limit_train_batches", default=1.0, type=int)
    group.add_argument("--limit_val_batches", default=1.0, type=int)

    parser.add_argument("--log-gradients", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpus", type=int, nargs='+', default=None, help="List of GPUs to use for training")

    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


def main(args):
    log_settings = wandb.Settings(start_method="thread")
    log_dir = os.environ["AEGNN_LOG_DIR"]
    loggers = [aegnn.utils.loggers.LoggingLogger(None if args.debug else log_dir, name="debug")]
    project = f"aegnn-{args.dataset}-{args.task}"
    experiment_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()
    
    model = None

    if args.checkpoint is not None:
        from aegnn.utils.callbacks.model_io import load_model

        model = load_model(args.checkpoint)
    else:
        model = aegnn.models.by_task(args.task)(args.model, args.dataset, num_classes=dm.num_classes,
                                            img_shape=dm.dims, dim=args.dim, bias=True, root_weight=True)

    if not args.debug:
        wandb_logger = pl.loggers.WandbLogger(project=project, save_dir=log_dir, settings=log_settings)
        if args.log_gradients:
            wandb_logger.watch(model, log="gradients")  # gradients plot every 100 training batches
        loggers.append(wandb_logger)
    logger = pl.loggers.LoggerCollection(loggers)

    checkpoint_path = os.path.join(log_dir, "checkpoints", args.dataset, args.task, experiment_name)
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        aegnn.utils.callbacks.BBoxLogger(classes=dm.classes),
        aegnn.utils.callbacks.PHyperLogger(args),
        aegnn.utils.callbacks.EpochLogger(),
        aegnn.utils.callbacks.FileLogger([model, model.model, dm]),
        aegnn.utils.callbacks.FullModelCheckpoint(dirpath=checkpoint_path)
    ]

    trainer_kwargs = dict()
    #
        # Set GPU configuration
    if args.gpus is not None and len(args.gpus) > 1:
        # Use DDP for multi-GPU training
        trainer_kwargs["accelerator"] = "ddp"  # Corrected to use 'ddp' as the accelerator
    elif args.gpus is not None and len(args.gpus) == 1:
        # Single GPU training
        trainer_kwargs["accelerator"] = "gpu"
    else:
        # CPU training
        trainer_kwargs["accelerator"] = "cpu"

    trainer_kwargs["devices"] = args.gpus if args.gpus is not None else 1  # Use the specified GPUs or default to 1 device
    #
    trainer_kwargs["profiler"] = "simple" if args.profile else False
    trainer_kwargs["weights_summary"] = "full"
    trainer_kwargs["track_grad_norm"] = 2 if args.log_gradients else -1

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, **trainer_kwargs)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    arguments = parse_args()
    pl.seed_everything(arguments.seed)
    main(arguments)
