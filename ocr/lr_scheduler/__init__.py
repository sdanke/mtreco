from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts
from .one_cycle import OneCycle


__all__ = ['CyclicLR', 'OneCycle']


def get_lr_scheduler(optimizer, config):
    # if config['type'] == 'CyclicLR':
    #     return CyclicLR(optimizer, **config['params'])
    # elif config['type'] == 'OneCycle':
    #     return OneCycle(optimizer, **config['params'])
    # else:
    #     raise NotImplementedError
    global_vars = globals()
    if config['type'] in global_vars:
        return global_vars[config['type']](optimizer, **config['params'])
    else:
        raise NotImplementedError
