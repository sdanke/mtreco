import math


class OneCycle(object):
    """Sets the learning rate of each parameter group according to
    1Cycle learning rate policy (1CLR). 1CLR is a variation of the
    Cyclical Learning Rate (CLR) policy that involves one cycle followed by
    decay. The policy simultaneously cycles the learning rate (and momentum)
    between two boundaries with a constant frequency, as detailed in
    the paper `A disciplined approach to neural network hyper-parameters`_.

    1CLR policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This implementation was adapted from the github repo: `pytorch/pytorch`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        cycle_min_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        cycle_max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_lr - cycle_min_lr).
            The lr at any cycle is the sum of cycle_min_lr
            and some scaling of the amplitude; therefore
            cycle_max_lr may not actually be reached depending on
            scaling function.
        decay_lr_rate(float): Decay rate for learning rate. Default: 0.
        cycle_first_step_size (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        cycle_second_step_size (int): Number of training iterations in the
            decreasing half of a cycle. If cycle_second_step_size is None,
            it is set to cycle_first_step_size. Default: None
        cycle_first_stair_count(int): Number of stairs in first half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        cycle_second_stair_count(int): Number of stairs in second half of cycle phase. This means
        lr/mom are changed in staircase fashion. Default 0, means staircase disabled.
        decay_step_size (int): Intervals for applying decay in decay phase. Default: 0, means no decay.
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'cycle_min_mom' and 'cycle_max_mom'.
            Default: True
        cycle_min_mom (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        cycle_max_mom (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (cycle_max_mom - cycle_min_mom).
            The momentum at any cycle is the difference of cycle_max_mom
            and some scaling of the amplitude; therefore
            cycle_min_mom may not actually be reached depending on
            scaling function. Default: 0.9
        decay_mom_rate (float): Decay rate for momentum. Default: 0.
        last_batch_iteration (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_batch_iteration=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.OneCycle(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay: https://arxiv.org/abs/1803.09820
    """
    def __init__(self,
                 optimizer,
                 cycle_min_lr,
                 cycle_max_lr,
                 decay_lr_rate=0.,
                 cycle_first_step_size=2000,
                 cycle_second_step_size=None,
                 cycle_first_stair_count=0,
                 cycle_second_stair_count=None,
                 decay_step_size=0,
                 cycle_momentum=True,
                 cycle_min_mom=0.8,
                 cycle_max_mom=0.9,
                 decay_mom_rate=0.,
                 last_batch_iteration=-1):

        self.optimizer = optimizer

        # Initialize cycle shape
        self._initialize_cycle(cycle_first_step_size,
                               cycle_second_step_size,
                               cycle_first_stair_count,
                               cycle_second_stair_count,
                               decay_step_size)

        # Initialize cycle lr
        self._initialize_lr(self.optimizer,
                            cycle_min_lr,
                            cycle_max_lr,
                            decay_lr_rate,
                            last_batch_iteration)

        # Initialize cyclic momentum
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            self._initialize_momentum(self.optimizer,
                                      cycle_min_mom,
                                      cycle_max_mom,
                                      decay_mom_rate,
                                      last_batch_iteration)

        # Initalize batch iteration tracker
        self.last_batch_iteration = last_batch_iteration

    # Configure cycle shape
    def _initialize_cycle(self,
                          cycle_first_step_size,
                          cycle_second_step_size,
                          cycle_first_stair_count,
                          cycle_second_stair_count,
                          decay_step_size):
        cycle_first_step_size = float(cycle_first_step_size)
        cycle_second_step_size = float(
            cycle_second_step_size
        ) if cycle_second_step_size is not None else cycle_first_step_size

        self.total_size = cycle_first_step_size + cycle_second_step_size
        self.step_ratio = cycle_first_step_size / self.total_size
        self.first_stair_count = cycle_first_stair_count
        self.second_stair_count = cycle_first_stair_count if cycle_second_stair_count is None else cycle_second_stair_count
        self.decay_step_size = decay_step_size

    # Configure lr schedule
    def _initialize_lr(self,
                       optimizer,
                       cycle_min_lr,
                       cycle_max_lr,
                       decay_lr_rate,
                       last_batch_iteration):
        self.min_lrs = [cycle_min_lr] * len(optimizer.param_groups)
        if last_batch_iteration == -1:
            for lr, group in zip(self.min_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = [cycle_max_lr] * len(optimizer.param_groups)
        self.decay_lr_rate = decay_lr_rate

    # Configure momentum schedule
    def _initialize_momentum(self,
                             optimizer,
                             cycle_min_mom,
                             cycle_max_mom,
                             decay_mom_rate,
                             last_batch_iteration):
        if 'betas' not in optimizer.defaults:
            optimizer_name = type(optimizer).__name__
            print(
                f"cycle_momentum is disabled because optimizer {optimizer_name} does not support momentum, no betas attribute in defaults"
            )
            self.cycle_momentum = False
            return

        self.decay_mom_rate = decay_mom_rate
        self.min_moms = [(cycle_min_mom, 0.99)] * len(optimizer.param_groups)
        self.max_moms = [(cycle_max_mom, 0.99)] * len(optimizer.param_groups)

        if last_batch_iteration == -1:
            for momentum, group in zip(self.min_moms, optimizer.param_groups):
                group['betas'] = momentum

    def _get_scale_factor(self):
        batch_iteration = (self.last_batch_iteration + 1)
        cycle = math.floor(1 + batch_iteration / self.total_size)
        x = 1. + batch_iteration / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        return scale_factor

    def _get_cycle_mom(self):
        scale_factor = self._get_scale_factor()
        momentums = []
        for base_betas, max_betas in zip(self.min_moms, self.max_moms):
            cycle_min_mom = base_betas[0]
            cycle_max_mom = max_betas[0]
            base_height = (cycle_max_mom - cycle_min_mom) * scale_factor
            momentum = cycle_max_mom - base_height
            momentums.append((momentum, base_betas[1]))
        return momentums

    def _get_cycle_lr(self):
        scale_factor = self._get_scale_factor()
        lrs = []
        for cycle_min_lr, cycle_max_lr in zip(self.min_lrs, self.max_lrs):
            base_height = (cycle_max_lr - cycle_min_lr) * scale_factor
            lr = cycle_min_lr + base_height
            lrs.append(lr)

        return lrs

    def _get_decay_mom(self, decay_batch_iteration):
        decay_interval = decay_batch_iteration / self.decay_step_size
        mom_decay_factor = (1 + self.decay_mom_rate * decay_interval)
        momentums = [(beta0 * mom_decay_factor, beta1) for beta0, beta1 in self.max_moms]
        return momentums

    def _get_decay_lr(self, decay_batch_iteration):
        """Calculates the learning rate at batch index. This function is used
        after the cycle completes and post cycle decaying of lr/mom is enabled.
        This function treats `self.last_batch_iteration` as the last batch index.
        """
        decay_interval = decay_batch_iteration / self.decay_step_size
        lr_decay_factor = (1 + self.decay_lr_rate * decay_interval)
        lrs = [cycle_min_lr / lr_decay_factor for cycle_min_lr in self.min_lrs]

        return lrs

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_batch_iteration` as the last batch index.
        """
        if self.last_batch_iteration < self.total_size:
            return self._get_cycle_lr()
        return self._get_decay_lr(self.last_batch_iteration - self.total_size + 1)

    def get_mom(self):
        """Calculates the momentum at batch index. This function treats
        `self.last_batch_iteration` as the last batch index.
        """
        if not self.cycle_momentum:
            return None

        if self.last_batch_iteration < self.total_size:
            return self._get_cycle_mom()
        return self._get_decay_mom(self.last_batch_iteration - self.total_size + 1)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, batch_iteration=None):
        """ Updates the optimizer with the learning rate for the last batch index.
        `self.last_batch_iteration` is treated as the last batch index.

        If self.cycle_momentum is true, also updates optimizer momentum.
        """
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1

        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        if self.cycle_momentum:
            momentums = self.get_mom()
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['betas'] = momentum

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']
