import logging
import os
import pytorch_lightning as pl
import torch


class FullModelCheckpoint(pl.callbacks.ModelCheckpoint):
    FILE_EXTENSION = ".pt"

    def _save_model(self, trainer: pl.Trainer, filepath: str) -> None:
        trainer.dev_debugger.track_checkpointing_history(filepath)
        if trainer.should_rank_save_checkpoint:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Extract the underlying model from DistributedDataParallel if it is wrapped
            model_to_save = trainer.model.module if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel) else trainer.model
            
            import dill
            with open(filepath, 'wb') as f:
                dill.dump(model_to_save, f)
            logging.debug(f"Save model checkpoint @ {filepath}")
            
        # Ensure all processes wait until the checkpoint is saved
        trainer.accelerator.barrier()
