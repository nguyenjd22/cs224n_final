import os
from copy import deepcopy
import random, torch, numpy
from src.systems import systems
from src.utils.utils import load_json
from src.utils.setup import process_config
import pytorch_lightning as pl

SYSTEM = {
    'PretrainSystem': systems.PretrainSystem,
    'PretrainTwoViewsSystem': systems.PretrainTwoViewsSystem,
    'TransferSystem': systems.TransferSystem,
}


def run(config_path, gpu_device=-1):
    config = process_config(config_path)
    if gpu_device >= 0: config.gpu_device = gpu_device
    seed_everything(config.seed, use_cuda=config.cuda)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=2500,
        period=1,
    )
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=([config.gpu_device] if config.cuda else None),
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
        #limit_val_batches=2,
        val_check_interval=0.5,
        profiler=True,
    )
    trainer.fit(system)


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()
    run(args.config, gpu_device=args.gpu_device)

