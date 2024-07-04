import hydra

import torch
import pytorch_lightning as pl

from utils.import_utils import instantiate

@hydra.main(version_base=None, config_path="conf", config_name="pretrain_mask-based_MVDR")
def main(cfg):
    torch.cuda.empty_cache()
    pl.seed_everything(0)

    dm = instantiate(cfg.data_module)

    model = instantiate(cfg.lightning_module)
    if hasattr(cfg, "ckpt_path"):
        model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])

    if not cfg.debug:
        logger = instantiate(cfg.logger)
        callbacks = [instantiate(cb) for cb in cfg.callbacks]
        trainer = pl.Trainer(
            logger=logger, 
            callbacks=callbacks, 
            **cfg.trainer
        )
    else:
        print("Debug mode.")
        trainer = pl.Trainer(
            logger=False, 
            enable_checkpointing=False, 
            callbacks=[], 
            **cfg.trainer
        )
    
    if not cfg.testonly:
        trainer.fit(model, dm)
        trainer.test(dataloaders = dm)
    else:
        trainer.test(model, dm)

if __name__ == "__main__":
    main()
