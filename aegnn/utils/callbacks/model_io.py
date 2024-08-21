import dill
import torch
import pytorch_lightning as pl
from torch.nn.parallel import DistributedDataParallel

def save_model(trainer, filepath):
    from aegnn.models import RecognitionModel, DetectionModel
    
    # Start with the full model
    model_to_save = trainer.model

    # Unwrap from DistributedDataParallel if necessary
    if isinstance(model_to_save, DistributedDataParallel):
        model_to_save = model_to_save.module  # This removes the DDP wrapper

    # Unwrap from LightningDistributedModule if necessary
    # We need to check if we're dealing with the specific LightningDistributedModule class
    if type(model_to_save).__name__ == 'LightningDistributedModule':
        model_to_save = model_to_save.module  # Unwrap the LightningDistributedModule

    # Now, model_to_save should be the RecognitionModel or DetectionModel itself
    with open(filepath, 'wb') as f:
        dill.dump(model_to_save, f)

    print("Model saved successfully!")

def load_model(filepath):
    with open(filepath, 'rb') as f:
        core_model = dill.load(f)  # Load the GraphRes model
    print("Model loaded successfully!")
    
    return core_model
