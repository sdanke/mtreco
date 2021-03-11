import torch
import shutil
import os

from tensorboardX import SummaryWriter
from ocr.utils.io import load_model, save_model, load_json_file
from ocr.utils.logger import get_logger
from ocr.lr_scheduler import get_lr_scheduler


class BaseTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        if opt.with_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.logger = get_logger(opt.log)
        self._build_model()
        self._build_optimizer()
        self._build_criterion()
        self._build_converter()
        self._build_dataloader()
        self._build_summary_writer()
        self.client_state = {'epoch': 0, 'step': 0}
        if not opt.fixed_lr:
            self.scheduler = get_lr_scheduler(self.optimizer, load_json_file(opt.lr_scheduler_config))
        else:
            self.scheduler = None

    def _build_optimizer(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _build_converter(self):
        raise NotImplementedError

    def _build_dataloader(self):
        raise NotImplementedError

    def _build_criterion(self):
        raise NotImplementedError

    def _build_summary_writer(self):
        opt = self.opt
        self.writer = SummaryWriter(log_dir=opt.exp_dir)

    def train_batch(self, step, batch_data):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def resume(self):
        opt = self.opt
        model_path = os.path.join(opt.exp_dir, 'model_last.pth')
        if os.path.exists(model_path):
            self.client_state = load_model(self.model, model_path, self.optimizer, self.scheduler, resume=True)
        self.logger.info(f'Resumed from: {model_path}')

    def save(self, epoch, step, phase='last'):
        self.client_state['step'] = step
        self.client_state['epoch'] = epoch
        model_path = os.path.join(self.opt.exp_dir, f'model_{phase}.pth')
        if os.path.exists(model_path):
            shutil.copy(model_path, f'{model_path}.bk')
        if phase == 'last':
            save_model(model_path, self.model, self.client_state, self.optimizer, self.scheduler)
        else:
            save_model(model_path, self.model, self.client_state)

    def find_lr(self, config, num_iters):
        raise NotImplementedError
