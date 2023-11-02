from omegaconf import OmegaConf
import torch
from contextlib import suppress

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def get_autocast(precision):
    # assert not (use_xla() and 'amp' in precision),\
    #     'currently pytorch xla does not support amp training!'
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
def load_config(*yaml_files, cli_args=[], extra_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    yaml_confs += [OmegaConf.from_cli(extra_args)]
    conf = OmegaConf.merge(*yaml_confs, cli_args)
    OmegaConf.resolve(conf)
    return conf

def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)