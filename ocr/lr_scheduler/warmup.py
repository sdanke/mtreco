import math
from torch.optim import Optimizer


class WarmupLR(object):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then fix at max lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = torch.optim.WarmupLR(optimizer)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 last_batch_iteration: int = -1):

        self.optimizer = optimizer

        self.min_lrs = self._format_param(self.optimizer, warmup_min_lr, "min_lr")
        self.max_lrs = self._format_param(self.optimizer, warmup_max_lr, "max_lr")
        self.delta_lrs = [big - small for big, small in zip(self.max_lrs, self.min_lrs)]
        self.warmup_num_steps = warmup_num_steps
        self.inverse_log_warm_up = 1.0 / math.log(warmup_num_steps)
        self.last_batch_iteration = last_batch_iteration

    def get_lr(self):
        if self.last_batch_iteration < 0:
            print(
                "Attempting to get learning rate from scheduler before it has started")
            return [0.0]
        gamma = self._get_gamma()
        return [
            min_lr + (delta_lr * gamma) for min_lr,
            delta_lr in zip(self.min_lrs,
                            self.delta_lrs)
        ]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
        return 1.0

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(
                    len(optimizer.param_groups),
                    param_name,
                    FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)


class WarmupDecayLR(WarmupLR):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then decay at linear rate over the remaining training steps.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_num_steps (int): total number of training steps
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = WarmupDecayLR(optimizer, 1000000)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """
    def __init__(self,
                 optimizer: Optimizer,
                 total_num_steps: int,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 last_batch_iteration: int = -1):

        self.total_num_steps = total_num_steps
        super(WarmupDecayLR,
              self).__init__(optimizer,
                             warmup_min_lr,
                             warmup_max_lr,
                             warmup_num_steps,
                             last_batch_iteration)
        if self.total_num_steps < self.warmup_num_steps:
            print('total_num_steps {} is less than warmup_num_steps {}'.format(
                total_num_steps,
                warmup_num_steps))

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
        return max(
            0.0,
            float(self.total_num_steps - self.last_batch_iteration) /
            float(max(1.0,
                      self.total_num_steps - self.warmup_num_steps)))
