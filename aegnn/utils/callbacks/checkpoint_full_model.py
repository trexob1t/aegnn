import logging
import os
import pytorch_lightning as pl
import torch


class FullModelCheckpoint(pl.callbacks.ModelCheckpoint):
    FILE_EXTENSION = ".pt"

    def _save_model(self, trainer: pl.Trainer, filepath: str) -> None:
        trainer.dev_debugger.track_checkpointing_history(filepath)
        if trainer.should_rank_save_checkpoint and trainer.global_rank == 0:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

        trainer.accelerator.barrier() # Ensure all gpus wait until directory is made and then save model

        torch.save(trainer.model.state_dict(), filepath)
        logging.debug(f"Save model checkpoint @ {filepath}")
