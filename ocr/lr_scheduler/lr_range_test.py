import math
from torch.optim import Optimizer


class LRRangeTest(object):
    """Sets the learning rate of each parameter group according to
    learning rate range test (LRRT) policy. The policy increases learning
    rate starting from a base value with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters: Part1`_.

    LRRT policy is used for finding maximum LR that trains a model without divergence, and can be used to
    configure the LR boundaries for Cylic LR schedules.

    LRRT changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_range_test_min_lr (float or list): Initial learning rate which is the
            lower boundary in the range test for each parameter group.
        lr_range_test_step_size (int): Interval of training steps to increase learning rate. Default: 2000
        lr_range_test_step_rate (float): Scaling rate for range test. Default: 1.0
        lr_range_test_staircase (bool): Scale in staircase fashion, rather than continous. Default: False.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.LRRangeTest(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

        _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay:
        https://arxiv.org/abs/1803.09820
"""
    def __init__(self,
                 optimizer: Optimizer,
                 lr_range_test_min_lr: float = 1e-3,
                 lr_range_test_step_size: int = 2000,
                 lr_range_test_step_rate: float = 1.0,
                 lr_range_test_staircase: bool = False,
                 last_batch_iteration: int = -1):

        self.optimizer = optimizer

        if isinstance(lr_range_test_min_lr,
                      list) or isinstance(lr_range_test_min_lr,
                                          tuple):
            if len(lr_range_test_min_lr) != len(self.optimizer.param_groups):
                raise ValueError("expected {} lr_range_test_min_lr, got {}".format(
                    len(self.optimizer.param_groups),
                    len(lr_range_test_min_lr)))
            self.min_lr = list(lr_range_test_min_lr)
        else:
            self.min_lr = [lr_range_test_min_lr] * len(self.optimizer.param_groups)

        self.step_size = lr_range_test_step_size
        self.step_rate = lr_range_test_step_rate
        self.last_batch_iteration = last_batch_iteration
        self.staircase = lr_range_test_staircase
        self.interval_fn = self._staircase_interval if lr_range_test_staircase else self._continous_interval

        if last_batch_iteration == -1:
            self._update_optimizer(self.min_lr)

    def _staircase_interval(self):
        return math.floor(float(self.last_batch_iteration + 1) / self.step_size)

    def _continous_interval(self):
        return float(self.last_batch_iteration + 1) / self.step_size

    def _get_increase(self):
        return (1 + self.step_rate * self.interval_fn())

    def get_lr(self):
        lr_increase = self._get_increase()
        return [
            lr_range_test_min_lr * lr_increase for lr_range_test_min_lr in self.min_lr
        ]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def _update_optimizer(self, group_lrs):
        for param_group, lr in zip(self.optimizer.param_groups, group_lrs):
            param_group['lr'] = lr

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        self._update_optimizer(self.get_lr())
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']
