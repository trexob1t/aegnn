#import dill
import torch
import pytorch_lightning as pl
from torch.nn.parallel import DistributedDataParallel

def save_model(trainer, filepath):
    trainer.save_checkpoint(filepath)

    print("Model saved successfully!")

def load_model(args, dm):
    import aegnn
    filepath = args.checkpoint

    model = aegnn.models.by_task(args.task)(args.model, args.dataset, num_classes=dm.num_classes,
                                            img_shape=dm.dims, dim=args.dim, bias=True, root_weight=True)
    try:
        core_model = model.load_from_checkpoint(filepath)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        core_model = model
    
    return core_model
