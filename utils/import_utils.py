from importlib import import_module


def import_and_getattr(name):
    module_name, attr_name = name.rsplit(".", 1)
    return getattr(import_module(module_name), attr_name)


def instantiate(cfg, **kwargs):
    Cls = import_and_getattr(cfg._target_)
    del cfg._target_
    return Cls(**cfg, **kwargs)


# def load_from_checkpoint(cfg, ckpt_path, **kwargs):
#     Cls = import_and_getattr(cfg._target_)
#     del cfg._target_
#     return Cls.load_from_checkpoint(ckpt_path)